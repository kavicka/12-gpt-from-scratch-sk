"""The numbers a GPT has to beat, measured on the same validation tokens.

- uniform: every one of the 8192 tokens equally likely → loss = ln 8192 = 9.01
- unigram: token frequencies from train (ignores context completely)
- bigram: P(next | previous token) from train counts, add-k smoothing (one token of context)

Run:  uv run python baselines.py   → results_baselines.json
"""

import json
import math
from pathlib import Path

import numpy as np

V = 8192
K = 0.01  # add-k smoothing, fixed up front (not tuned on val)


def main() -> None:
    train = np.fromfile("data/train.bin", dtype=np.uint16).astype(np.int64)
    val = np.fromfile("data/val.bin", dtype=np.uint16).astype(np.int64)
    uni = np.bincount(train, minlength=V) + K
    uni = uni / uni.sum()
    unigram = float(-np.log(uni[val]).mean())
    pairs = np.bincount(train[:-1] * V + train[1:], minlength=V * V).reshape(V, V).astype(np.float64)
    pairs += K
    pairs /= pairs.sum(1, keepdims=True)
    bigram = float(-np.log(pairs[val[:-1], val[1:]]).mean())
    res = {
        "uniform": {"val_loss": math.log(V), "ppl": float(V)},
        "unigram": {"val_loss": unigram, "ppl": math.exp(unigram)},
        "bigram": {"val_loss": bigram, "ppl": math.exp(bigram), "add_k": K},
        "val_tokens": len(val),
    }
    Path("results_baselines.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
