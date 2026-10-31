"""Byte-level BPE tokenizer written from scratch (the same idea as GPT-2's, trained on Slovak).

1. Pre-split text with the GPT-2 regex into "chunks" (a word with its leading space, a number, punctuation).
   Merges never cross a chunk border, so "mesto." never becomes one token with the dot.
2. Every chunk starts as its UTF-8 bytes (256 possible symbols): any text can be encoded, no <unk>.
   "š" is 2 bytes in UTF-8, which is exactly why an English tokenizer wastes tokens on Slovak.
3. Training: count all adjacent pairs, merge the most frequent pair into a new symbol, repeat.
   Counting is incremental: after a merge only the chunks that contained the pair are recounted,
   and a heap finds the next most frequent pair. Naively recounting everything would take hours.

Run:  uv run python bpe.py   → models/bpe_sk.json (train on a sample of data/sk_train.txt)
"""

import heapq
import json
import time
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import regex as re

GPT2_SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
EOT = "<|endoftext|>"
VOCAB_SIZE = 8192  # 256 bytes + 7935 merges + 1 special token
TRAIN_SAMPLE_CHARS = 40_000_000


class BPE:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = merges
        self.ranks = {pair: 256 + i for i, pair in enumerate(merges)}  # pair → new token id (= its rank)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in self.ranks.items():
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.eot = 256 + len(merges)
        self.vocab_size = self.eot + 1
        self._cache: dict[str, list[int]] = {}

    # --- encoding ------------------------------------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = self._cache.get(chunk)
        if ids is not None:
            return ids
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # apply the merge that was learned EARLIEST among the pairs present, exactly as in training
            pair = min(pairwise(ids), key=lambda p: self.ranks.get(p, 1 << 30))
            new = self.ranks.get(pair)
            if new is None:
                break
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) < 2_000_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out = []
        for part_i, part in enumerate(text.split(EOT)):
            if part_i:
                out.append(self.eot)
            for chunk in GPT2_SPLIT.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(EOT.encode() if i == self.eot else self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        return EOT if i == self.eot else self.vocab[i].decode("utf-8", errors="replace")

    # --- save / load ----------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"merges": self.merges, "split": GPT2_SPLIT.pattern, "eot": EOT}))

    @classmethod
    def load(cls, path: Path) -> "BPE":
        return cls([tuple(m) for m in json.loads(path.read_text())["merges"]])


def train(text: str, vocab_size: int = VOCAB_SIZE, verbose: int = 500) -> BPE:
    n_merges = vocab_size - 256 - 1
    chunk_freq = Counter()
    for part in text.split(EOT):
        chunk_freq.update(GPT2_SPLIT.findall(part))
    words = [list(c.encode("utf-8")) for c in chunk_freq]
    freqs = list(chunk_freq.values())
    print(f"{sum(freqs):,} chunks, {len(words):,} unique")

    counts: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair → indices of words containing it
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for p in pairwise(w):
            counts[p] += f
            where[p].add(wi)
    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)

    merges, t0 = [], time.time()
    while len(merges) < n_merges and heap:
        neg, pair = heapq.heappop(heap)
        if counts.get(pair, 0) != -neg or -neg <= 0:
            continue  # stale heap entry: the count changed since it was pushed
        new = 256 + len(merges)
        merges.append(pair)
        changed = set()
        for wi in where.pop(pair):
            w, f = words[wi], freqs[wi]
            for p in pairwise(w):  # remove this word's old pairs …
                counts[p] -= f
                changed.add(p)
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                    out.append(new)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for p in pairwise(out):  # … and add its new ones
                counts[p] += f
                where[p].add(wi)
                changed.add(p)
        counts.pop(pair, None)
        for p in changed:
            if counts.get(p, 0) > 0:
                heapq.heappush(heap, (-counts[p], p))
        if verbose and len(merges) % verbose == 0:
            tok = BPE(merges).vocab[new].decode("utf-8", errors="replace")
            print(f"merge {len(merges):5d}: {tok!r:>16} (count {-neg:,})  {time.time() - t0:.0f} s", flush=True)
    return BPE(merges)


def main() -> None:
    text = Path("data/sk_train.txt").read_text(encoding="utf-8")[:TRAIN_SAMPLE_CHARS]
    t0 = time.time()
    bpe = train(text)
    print(f"trained {len(bpe.merges)} merges in {time.time() - t0:.0f} s")
    bpe.save(Path("models/bpe_sk.json"))
    sample = "Bratislava je hlavné mesto Slovenska a leží na Dunaji."
    ids = bpe.encode(sample)
    assert bpe.decode(ids) == sample
    print(len(ids), [bpe.token_str(i) for i in ids])


if __name__ == "__main__":
    main()
