"""Turn the results_*.json files into the plots and the GIF used in the README."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

BLUE, RED, GREEN, GRAY, ORANGE, DARK = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c", "#1e293b"
SIZE_COLORS = {"tiny": "#93c5fd", "small": "#3b82f6", "base": "#1e3a8a"}
Path("assets").mkdir(exist_ok=True)


def load(name: str) -> dict:
    return json.loads(Path(name).read_text())


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def tokenizer() -> None:
    r = load("results_tokenizer.json")
    names = {
        "mine_8k": "my BPE\n8,192 tokens, Slovak",
        "gpt2_50k": "GPT-2\n50,257 tokens",
        "o200k_gpt4o": "GPT-4o (o200k)\n~200k tokens",
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    vals = [r["rows"][k]["tokens_per_word"] for k in names]
    bars = a1.bar(list(names.values()), vals, color=[GREEN, RED, BLUE])
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
    a1.set(ylabel="tokens per word", title=f"Same Slovak text ({r['words']:,} words of validation set)")
    a1.grid(alpha=0.3, axis="y")
    a2.axis("off")
    y = 0.95
    for key, color in (("gpt2_50k", RED), ("o200k_gpt4o", BLUE), ("mine_8k", GREEN)):
        toks = r["examples"][key][0]
        a2.text(
            0,
            y,
            f"{names[key].splitlines()[0]}: {len(toks)} tokens",
            color=color,
            fontsize=10,
            weight="bold",
            transform=a2.transAxes,
        )
        line = " | ".join(t.replace(" ", "·") for t in toks)
        for i, part in enumerate(textwrap.wrap(line, 92)):
            a2.text(0, y - 0.07 * (i + 1), part, fontsize=8.5, family="monospace", transform=a2.transAxes)
        y -= 0.07 * (len(textwrap.wrap(line, 92)) + 1) + 0.04
    a2.set_title("One sentence, split into tokens (· = space, � = half of a UTF-8 character)", fontsize=10)
    save(fig, "tokenizer.png")


def loss_curves() -> None:
    base = load("results_baselines.json")
    runs = {
        s: load(f"results_train_{s}.json")
        for s in ("tiny", "small", "base")
        if Path(f"results_train_{s}.json").exists()
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for s, r in runs.items():
        ev = r["evals"]
        a1.plot(
            [e["tokens"] / 1e6 for e in ev],
            [e["val_loss"] for e in ev],
            "-",
            color=SIZE_COLORS[s],
            lw=2,
            label=f"{s}: {r['params_non_embedding'] / 1e6:.1f}M params (+{(r['params_total'] - r['params_non_embedding']) / 1e6:.1f}M emb.)",
        )
    for k, ls in (("unigram", ":"), ("bigram", "--")):
        a1.axhline(base[k]["val_loss"], color=GRAY, ls=ls, label=f"{k} baseline ({base[k]['val_loss']:.2f})")
    a1.set(
        xlabel="training tokens (millions)",
        ylabel="validation loss (nats / token)",
        ylim=(2.5, 8),
        title="Validation loss while training",
    )
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    p = [r["params_non_embedding"] for r in runs.values()]
    v = [r["final_val_loss"] for r in runs.values()]
    a2.plot(p, v, "o-", color=DARK)
    for s, x, y in zip(runs, p, v):
        a2.annotate(f"{s}\nppl {np.exp(y):.0f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    a2.set(
        xscale="log",
        xlabel="parameters without embeddings",
        ylabel="final validation loss",
        title="Same data and steps, bigger model",
    )
    a2.grid(alpha=0.3, which="both")
    save(fig, "loss.png")


def kv_cache() -> None:
    g = load("results_generate.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for key, color, label in (
        ("no_cache", RED, "no cache: whole sequence again"),
        ("cache", GREEN, "KV cache: only the new token"),
    ):
        rows = np.array(g["cost_vs_length_ms"][key])
        a1.plot(rows[:, 0], rows[:, 1], color=color, lw=1, alpha=0.8, label=label)
    a1.set(
        xlabel="tokens already in the sequence",
        ylabel="ms for one new token (CPU)",
        title="Cost of one generation step",
    )
    a1.grid(alpha=0.3)
    a1.legend(fontsize=9)
    kv = g["kv_cache"]
    devs = list(kv)
    x = np.arange(len(devs))
    a2.bar(x - 0.18, [kv[d]["tokens_per_s_no_cache"] for d in devs], 0.36, color=RED, label="no cache")
    a2.bar(x + 0.18, [kv[d]["tokens_per_s_cache"] for d in devs], 0.36, color=GREEN, label="KV cache")
    for i, d in enumerate(devs):
        a2.text(
            i + 0.18, kv[d]["tokens_per_s_cache"], f"{kv[d]['speedup']:.1f}×", ha="center", va="bottom", fontsize=10
        )
    a2.set_xticks(x, [d.upper() for d in devs])
    a2.set(ylabel="tokens / second", title=f"Generating {kv[devs[0]]['new_tokens']} tokens (greedy)")
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3, axis="y")
    save(fig, "kv_cache.png")


def attention() -> None:
    a = load("results_generate.json")["attention"]
    heads = a["heads"]
    picks = [
        ("looks at the previous token", max(heads, key=lambda h: h["prev_token"])),
        ("parks on the first token", max(heads, key=lambda h: h["first_token"])),
        (
            "last layer, head 0",
            next(h for h in heads if h["layer"] == max(x["layer"] for x in heads) and h["head"] == 0),
        ),
    ]
    toks = [t.replace(" ", "·") for t in a["tokens"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for ax, (title, h) in zip(axes, picks):
        m = np.array(a["maps"][h["layer"]][h["head"]])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(toks)), toks, rotation=90, fontsize=7)
        ax.set_yticks(range(len(toks)), toks, fontsize=7)
        ax.set_title(f"layer {h['layer']}, head {h['head']}: {title}", fontsize=10)
        ax.set_xlabel("key (looked at)")
    axes[0].set_ylabel("query (the token that is predicting)")
    save(fig, "attention.png")


def generation_gif() -> None:
    steps = load("results_generate.json")["gif_steps"]
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(8, 5.6), gridspec_kw={"height_ratios": [1, 1.4]})

    def frame(i):
        s = steps[min(i, len(steps) - 1)]
        ax_t.clear()
        ax_t.axis("off")
        ax_t.text(0, 0.85, "\n".join(textwrap.wrap(s["text"], 60)), fontsize=12, va="top", family="monospace")
        ax_t.set_title(f"token {min(i, len(steps) - 1) + 1}: the model picks the next token (T = 0.7)", fontsize=10)
        ax_b.clear()
        labels = [t.replace(" ", "·") for t, _ in s["top"]][::-1]
        probs = [p for _, p in s["top"]][::-1]
        colors = [GREEN if t == s["chosen"] else GRAY for t, _ in s["top"]][::-1]
        ax_b.barh(labels, probs, color=colors)
        ax_b.set(xlim=(0, 1), xlabel="probability of the next token (top 6)")
        ax_b.text(
            0.98, 0.05, f"drawn: {s['chosen'].replace(' ', '·')!r}", transform=ax_b.transAxes, ha="right", color=GREEN
        )
        return []

    fig.tight_layout()
    FuncAnimation(fig, frame, frames=len(steps) + 6).save("assets/generation.gif", writer=PillowWriter(fps=2), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    tokenizer()
    loss_curves()
    kv_cache()
    attention()
    generation_gif()
