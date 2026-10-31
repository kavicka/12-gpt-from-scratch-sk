"""Download a slice of Slovak Wikipedia and split it into train / val text files.

Source: the `wikimedia/wikipedia` dataset on Hugging Face, dump 2023-11-01, config `20231101.sk`
(one Parquet file of 240 MB, plain article text already stripped of wiki markup).

Run:  uv run python data.py   → data/sk_train.txt, data/sk_val.txt, data/stats.json
"""

import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.sk/train-00000-of-00001.parquet"
DATA = Path("data")
MAX_CHARS = 300_000_000  # ~300 MB of text is plenty for a 10–30M parameter model
MIN_CHARS = 500  # skip stubs ("X is a village in Y."), they repeat one template thousands of times
VAL_SHARE = 0.01
EOT = "\n<|endoftext|>\n"  # document separator, becomes one special token


def download() -> Path:
    path = DATA / "skwiki-00000.parquet"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        print("downloading", URL)
        urllib.request.urlretrieve(URL, path)
    return path


def main() -> None:
    table = pq.read_table(download(), columns=["title", "text"])
    texts = [t.strip() for t in table.column("text").to_pylist()]
    n_all = len(texts)
    texts = [t for t in texts if len(t) >= MIN_CHARS]
    # shuffle whole articles (seeded) so val is not one alphabetical block of titles
    random.Random(0).shuffle(texts)
    keep, total = [], 0
    for t in texts:
        if total >= MAX_CHARS:
            break
        keep.append(t)
        total += len(t)
    n_val = max(1, int(len(keep) * VAL_SHARE))
    val, train = keep[:n_val], keep[n_val:]
    (DATA / "sk_train.txt").write_text(EOT.join(train), encoding="utf-8")
    (DATA / "sk_val.txt").write_text(EOT.join(val), encoding="utf-8")
    stats = {
        "source": URL,
        "articles_in_file": n_all,
        "articles_min_chars": MIN_CHARS,
        "articles_used": len(keep),
        "train_articles": len(train),
        "val_articles": len(val),
        "train_chars": sum(map(len, train)),
        "val_chars": sum(map(len, val)),
        "train_words": sum(len(t.split()) for t in train),
        "val_words": sum(len(t.split()) for t in val),
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
