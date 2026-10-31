"""Use the trained GPT: samples at several temperatures, the KV-cache speed-up, attention maps.

Run:  uv run python generate.py   → results_generate.json (needs models/gpt_base.pt, models/bpe_sk.json)
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.nn import functional as F

from bpe import BPE
from model import GPT, Config

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROMPTS = ["Bratislava je", "V roku 1993", "Poisťovňa"]
TEMPERATURES = [0.2, 0.7, 1.0, 1.5]
ATT_SENTENCE = "Mária kúpila psa, ale ten pes ju nepočúval, lebo Mária ho nekŕmila."


def load(size: str = "base", device=DEVICE) -> GPT:
    ck = torch.load(f"models/gpt_{size}.pt", map_location="cpu")
    model = GPT(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    return model.to(device).eval()


def samples(model: GPT, bpe: BPE) -> list[dict]:
    out = []
    for prompt in PROMPTS:
        for t in TEMPERATURES:
            g = torch.Generator(device="cpu").manual_seed(0)
            idx = torch.tensor([bpe.encode(prompt)])
            ids = model.to("cpu").generate(idx, 60, temperature=t, top_k=None, generator=g)[0].tolist()
            out.append({"prompt": prompt, "temperature": t, "text": bpe.decode(ids)})
    # top-k / top-p at T = 1.0: cut the long tail of unlikely tokens instead of sharpening everything
    for kw in ({"top_k": 50}, {"top_p": 0.9}):
        g = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.tensor([bpe.encode(PROMPTS[0])])
        ids = model.generate(idx, 60, temperature=1.0, generator=g, **kw)[0].tolist()
        out.append({"prompt": PROMPTS[0], "temperature": 1.0, **kw, "text": bpe.decode(ids)})
    return out


def sync(device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def kv_cache_benchmark(bpe: BPE, repeats: int = 3) -> dict:
    """Greedy generation of the same tokens with and without the cache. Same output, different cost."""
    res = {}
    for dev in [torch.device("cpu")] + ([DEVICE] if DEVICE.type != "cpu" else []):
        model = load(device=dev)
        prompt = torch.tensor([bpe.encode(PROMPTS[0])], device=dev)
        n_new = model.cfg.context - prompt.shape[1]
        outs, times = {}, {}
        for use_cache in (False, True):
            model.generate(prompt, 8, temperature=0, use_cache=use_cache)  # warm-up (kernels, allocator)
            ts = []
            for _ in range(repeats):
                sync(dev)
                t0 = time.perf_counter()
                o = model.generate(prompt, n_new, temperature=0, use_cache=use_cache)
                sync(dev)
                ts.append(time.perf_counter() - t0)
            outs[use_cache], times[use_cache] = o[0].tolist(), float(np.median(ts))
        res[dev.type] = {
            "new_tokens": n_new,
            "seconds_no_cache": times[False],
            "seconds_cache": times[True],
            "tokens_per_s_no_cache": n_new / times[False],
            "tokens_per_s_cache": n_new / times[True],
            "speedup": times[False] / times[True],
            "identical_output": outs[False] == outs[True],
        }
        print(dev.type, res[dev.type], flush=True)
    return res


def cost_vs_length(bpe: BPE) -> dict:
    """Time per generated token as the sequence grows: without a cache it grows, with a cache it stays flat."""
    model = load(device=torch.device("cpu"))
    prompt = torch.tensor([bpe.encode(PROMPTS[0])])
    rows = {}
    for use_cache in (False, True):
        caches = [{} for _ in model.blocks] if use_cache else None
        idx, x, per_token = prompt, prompt, []
        with torch.no_grad():
            while idx.shape[1] < model.cfg.context:
                t0 = time.perf_counter()
                logits, _ = model(x, caches=caches)
                nxt = logits[:, -1].argmax(-1, keepdim=True)
                per_token.append((idx.shape[1], (time.perf_counter() - t0) * 1000))
                idx = torch.cat([idx, nxt], 1)
                x = nxt if use_cache else idx
        rows["cache" if use_cache else "no_cache"] = per_token
    return rows


@torch.no_grad()
def attention(model: GPT, bpe: BPE) -> dict:
    model = model.to("cpu")
    ids = bpe.encode(ATT_SENTENCE)
    model(torch.tensor([ids]), keep_att=True)
    maps = [b.att.last_att[0].numpy() for b in model.blocks]  # per layer: (heads, T, T)
    T = len(ids)
    heads = []
    for li, m in enumerate(maps):
        for hi, a in enumerate(m):
            prev = float(np.mean([a[i, i - 1] for i in range(1, T)]))  # attention to the previous token
            first = float(np.mean(a[1:, 0]))  # attention parked on the first token ("no-op" sink)
            self_ = float(np.mean(np.diag(a)))
            heads.append({"layer": li, "head": hi, "prev_token": prev, "first_token": first, "self": self_})
    return {
        "sentence": ATT_SENTENCE,
        "tokens": [bpe.token_str(i) for i in ids],
        "maps": [m.round(4).tolist() for m in maps],
        "heads": heads,
    }


@torch.no_grad()
def next_token_steps(model: GPT, bpe: BPE, prompt: str = PROMPTS[0], n: int = 24, temperature: float = 0.7) -> list:
    """For the GIF: at every step the 6 most likely next tokens and which one was drawn."""
    model = model.to("cpu")
    g = torch.Generator().manual_seed(1)
    idx = torch.tensor([bpe.encode(prompt)])
    caches = [{} for _ in model.blocks]
    x, steps = idx, []
    for _ in range(n):
        logits, _ = model(x, caches=caches)
        p = F.softmax(logits[0, -1] / temperature, -1)
        top = torch.topk(p, 6)
        nxt = torch.multinomial(p, 1, generator=g)
        steps.append(
            {
                "text": bpe.decode(idx[0].tolist()),
                "top": [[bpe.token_str(int(i)), float(v)] for v, i in zip(top.values, top.indices)],
                "chosen": bpe.token_str(int(nxt)),
            }
        )
        idx = torch.cat([idx, nxt[None]], 1)
        x = nxt[None]
    return steps


def main() -> None:
    bpe = BPE.load(Path("models/bpe_sk.json"))
    model = load()
    res = {
        "device": str(DEVICE),
        "samples": samples(model, bpe),
        "kv_cache": kv_cache_benchmark(bpe),
        "cost_vs_length_ms": cost_vs_length(bpe),
        "attention": attention(model, bpe),
        "gif_steps": next_token_steps(model, bpe),
    }
    for s in res["samples"]:
        print(f"T={s['temperature']} {s.get('top_k', '')}{s.get('top_p', '')}: {s['text']!r}")
    top_prev = max(res["attention"]["heads"], key=lambda h: h["prev_token"])
    print("most 'previous-token' head:", top_prev)
    Path("results_generate.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
