"""Train the GPT on the encoded Slovak Wikipedia with my own training loop (AdamW, warmup + cosine).

Run:  uv run python train.py --size base --steps 6000    → models/gpt_base.pt, results_train_base.json
      uv run python train.py --size tiny --steps 6000    (the scaling comparison uses the same steps)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from model import GPT, SIZES

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LR = {"tiny": 2e-3, "small": 1e-3, "base": 6e-4}  # smaller models tolerate larger steps


def batches(split: str, batch: int, context: int, rng: np.random.Generator):
    data = np.memmap(f"data/{split}.bin", dtype=np.uint16, mode="r")
    while True:
        # random windows: the target is the same window shifted by one token
        ix = rng.integers(0, len(data) - context - 1, batch)
        x = np.stack([data[i : i + context] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1 : i + 1 + context] for i in ix]).astype(np.int64)
        yield torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup  # warmup: Adam's running averages are still noise at the start
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # cosine down to 10 % of the peak


@torch.no_grad()
def evaluate(model: GPT, val, iters: int) -> float:
    model.eval()
    losses = [model(*next(val))[1].item() for _ in range(iters)]
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", choices=list(SIZES))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = SIZES[args.size]
    model = GPT(cfg).to(DEVICE)
    peak = LR[args.size]
    # weight decay only on matrices; biases, LayerNorm gains and (tied) embeddings stay free
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "tok" not in n and "pos" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "tok" not in n and "pos" not in n)]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=peak,
        betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(args.seed)
    train, val = batches("train", args.batch, cfg.context, rng), batches("val", args.batch, cfg.context, rng)
    tokens_per_step = args.batch * cfg.context
    log = {
        "size": args.size,
        "config": cfg.to_dict(),
        "args": vars(args),
        "peak_lr": peak,
        "device": str(DEVICE),
        "params_total": model.n_params(non_embedding=False),
        "params_non_embedding": model.n_params(),
        "tokens_per_step": tokens_per_step,
        "train_tokens_available": len(np.memmap("data/train.bin", np.uint16)),
        "evals": [],
        "train_loss": [],
    }
    print(
        f"{args.size}: {log['params_total']:,} params ({log['params_non_embedding']:,} without embeddings) on {DEVICE}"
    )
    t0, running = time.time(), []
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate(model, val, args.eval_iters)
            tr = float(np.mean(running)) if running else None
            log["evals"].append(
                {
                    "step": step,
                    "tokens": step * tokens_per_step,
                    "val_loss": v,
                    "train_loss": tr,
                    "minutes": (time.time() - t0) / 60,
                }
            )
            print(
                f"step {step:5d}  val {v:.3f}  train {tr}  ppl {math.exp(v):7.1f}  {(time.time() - t0) / 60:5.1f} min",
                flush=True,
            )
            running = []
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, peak, args.warmup)
        x, y = next(train)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(loss.item())
        if step % 10 == 0:
            log["train_loss"].append([step, running[-1]])
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    log["minutes"] = (time.time() - t0) / 60
    log["tokens_per_second"] = args.steps * tokens_per_step / (time.time() - t0)
    log["final_val_loss"] = log["evals"][-1]["val_loss"]
    log["final_val_ppl"] = math.exp(log["final_val_loss"])
    log["epochs"] = args.steps * tokens_per_step / log["train_tokens_available"]
    Path("models").mkdir(exist_ok=True)
    torch.save({"config": cfg.to_dict(), "state": model.state_dict()}, f"models/gpt_{args.size}.pt")
    Path(f"results_train_{args.size}.json").write_text(json.dumps(log, indent=1))
    print(
        f"done in {log['minutes']:.1f} min, {log['tokens_per_second']:,.0f} tokens/s, val ppl {log['final_val_ppl']:.1f}"
    )


if __name__ == "__main__":
    main()
