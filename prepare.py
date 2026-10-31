"""Encode train / val text with my BPE into flat uint16 token arrays, plus the tokenizer comparison.

Tokens per word on the same Slovak validation text:
  my BPE (8192 tokens, trained on Slovak)  vs  GPT-2 (50,257, mostly English)  vs  GPT-4o's o200k (~200k, multilingual)

Run:  uv run python prepare.py   → data/train.bin, data/val.bin, results_tokenizer.json
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tiktoken

from bpe import BPE, EOT

DATA = Path("data")
SENTENCES = [
    "Bratislava je hlavné mesto Slovenska a leží na Dunaji.",
    "Najvyšším vrchom Vysokých Tatier je Gerlachovský štít.",
    "Poisťovňa uhradila škodu spôsobenú krupobitím na streche.",
]
_bpe: BPE | None = None


def _init() -> None:
    global _bpe
    _bpe = BPE.load(Path("models/bpe_sk.json"))


def _encode(block: str) -> list[int]:
    return _bpe.encode(block)


def encode_file(name: str, bpe: BPE, pool: Pool) -> np.ndarray:
    text = (DATA / f"{name}.txt").read_text(encoding="utf-8")
    docs = text.split(f"\n{EOT}\n")
    # every document ends with <|endoftext|>, so the model learns where articles stop
    parts = pool.map(_encode, docs, chunksize=64)
    ids = np.fromiter((i for p in parts for i in (*p, bpe.eot)), dtype=np.uint16)
    ids.tofile(DATA / f"{name.removeprefix('sk_')}.bin")
    return ids


def compare(text: str, mine: BPE) -> dict:
    words = len(text.split())
    rows = {}
    encoders = {
        "mine_8k": mine.encode,
        "gpt2_50k": tiktoken.get_encoding("gpt2").encode_ordinary,
        "o200k_gpt4o": tiktoken.get_encoding("o200k_base").encode_ordinary,
    }
    for name, enc in encoders.items():
        n = len(enc(text))
        rows[name] = {"tokens": n, "tokens_per_word": n / words, "chars_per_token": len(text) / n}
    examples = {}
    for name, enc_name in (("gpt2_50k", "gpt2"), ("o200k_gpt4o", "o200k_base")):
        enc = tiktoken.get_encoding(enc_name)
        examples[name] = [
            [enc.decode_single_token_bytes(t).decode("utf-8", "replace") for t in enc.encode(s)] for s in SENTENCES
        ]
    examples["mine_8k"] = [[mine.token_str(t) for t in mine.encode(s)] for s in SENTENCES]
    return {"words": words, "chars": len(text), "rows": rows, "sentences": SENTENCES, "examples": examples}


def main() -> None:
    bpe = BPE.load(Path("models/bpe_sk.json"))
    t0 = time.time()
    with Pool(initializer=_init) as pool:
        val = encode_file("sk_val", bpe, pool)
        train = encode_file("sk_train", bpe, pool)
    print(f"encoded train {len(train):,} / val {len(val):,} tokens in {time.time() - t0:.0f} s")
    val_text = (DATA / "sk_val.txt").read_text(encoding="utf-8").replace(f"\n{EOT}\n", "\n")
    res = compare(val_text, bpe)
    res.update(
        train_tokens=len(train),
        val_tokens=len(val),
        vocab_size=bpe.vocab_size,
        stats=json.loads((DATA / "stats.json").read_text()),
    )
    Path("results_tokenizer.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res["rows"], indent=1))


if __name__ == "__main__":
    main()
