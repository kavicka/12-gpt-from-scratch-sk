# Tutorial · 12 · GPT from scratch on Slovak

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial builds the project in the order you would write it: data → tokenizer → token files → baselines → the model → the training loop → sampling and the KV cache → attention maps. Each step explains **what** the code does, **why** it is written that way, and **how** it works. The README has all the numbers.

**Goal:** a byte-level BPE tokenizer and a decoder-only transformer (GPT), both written by hand, trained on Slovak Wikipedia on a Mac, measured against simple baselines.

**You need:** projects 04 (makemore: next-character prediction, training loop), 08 (cross-entropy, softmax, temperature) and 11 (PyTorch training on MPS). Libraries: PyTorch, `regex`, pyarrow, numpy; `tiktoken` only for the tokenizer comparison.

---

## Step 1 · Get the text ([`data.py`](data.py))

```python
MIN_CHARS = 500                    # skip one-line stubs
texts = [t for t in texts if len(t) >= MIN_CHARS]
random.Random(0).shuffle(texts)    # shuffle whole articles before splitting
val, train = keep[:n_val], keep[n_val:]     # 1 % of articles for validation
(DATA / "sk_train.txt").write_text(EOT.join(train))   # EOT = "\n<|endoftext|>\n"
```

**What:** downloads one 240 MB Parquet file of Slovak Wikipedia (dump 2023-11-01), keeps articles with at least 500 characters and splits them into 105,072 train and 1,061 validation articles (297M / 2.8M characters).

**Why skip short articles:** most of them are one-line village stubs built from the same template ("X is a village in district Y"). Thousands of copies would teach the model to repeat that template.

**Why shuffle whole articles:** the file is sorted by title. Without shuffling, validation would be one alphabetical block. Splitting by *article*, not by line, also keeps sentences of one article from landing on both sides.

**Why `<|endoftext|>`:** a separator between articles that later becomes one special token. The model learns where an article ends, and an article's start is not glued to the end of the previous one.

---

## Step 2 · Write your own BPE tokenizer ([`bpe.py`](bpe.py))

A model works with numbers, not letters. A **tokenizer** cuts text into pieces (tokens) and gives each piece an id. Characters would make sequences very long; whole words would need a huge vocabulary and could not handle new words. **BPE** (byte pair encoding) is in between: it starts from bytes and learns frequent pieces.

### 2a · Pre-split into chunks
```python
GPT2_SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```
The GPT-2 regex splits text into a word with its leading space, a number, punctuation or whitespace. `\p{L}` means "any letter in any language", so `š`, `ô`, `ľ` stay inside the word. **Why:** merges never cross a chunk border, so "mesto." never becomes one token together with the dot, and the vocabulary is spent on real word pieces.

### 2b · Start from bytes
```python
words = [list(c.encode("utf-8")) for c in chunk_freq]
```
Every chunk starts as its UTF-8 bytes, so there are 256 starting symbols and **nothing is ever unknown** (no `<unk>` token). Letters like `š` take 2 bytes; an English tokenizer that has not learned to merge them pays 2 tokens for one letter, which is why GPT-2 is expensive on Slovak.

### 2c · Training: merge the most frequent pair, again and again
```python
neg, pair = heapq.heappop(heap)
if counts.get(pair, 0) != -neg: continue      # stale heap entry, skip it
new = 256 + len(merges); merges.append(pair)
for wi in where.pop(pair):                    # only the chunks that contain this pair
    for p in pairwise(w): counts[p] -= f      # remove old pairs
    ... replace (a, b) with `new` ...
    for p in pairwise(out): counts[p] += f    # add new pairs
```
**How:** count every adjacent pair of symbols (weighted by how often the chunk appears), merge the most frequent pair into a new symbol, repeat 7,935 times → 256 + 7,935 + 1 special = 8,192 tokens.

**Why so complicated:** the naive version recounts all pairs in all chunks after every merge, which would take hours on 40M characters. Here:
- `where[pair]` remembers which chunks contain a pair, so only those are recounted;
- a **heap** gives the most frequent pair quickly. Instead of updating entries inside the heap, a new entry is pushed and old ones are skipped when popped if their count no longer matches ("lazy deletion").

Result: **43 s** on the Mac. On a small text it learns exactly the same 300 of 300 merges as the naive version, so the speed-up did not change the result.

**Why train on 40M characters and not all 297M:** pair frequencies stabilise long before that; more text only costs time.

### 2d · Encoding
```python
pair = min(pairwise(ids), key=lambda p: self.ranks.get(p, 1 << 30))
```
To encode a new chunk, apply the merge that was **learned earliest** among the pairs present, repeat until no pair is in the merge list. **Why in that order:** that is the order in which training built the tokens, so encoding reproduces them. Results are cached per chunk because Wikipedia repeats the same words millions of times.

---

## Step 3 · Encode everything and compare tokenizers ([`prepare.py`](prepare.py))

```python
parts = pool.map(_encode, docs, chunksize=64)                       # all CPU cores
ids = np.fromiter((i for p in parts for i in (*p, bpe.eot)), dtype=np.uint16)
ids.tofile(DATA / "train.bin")
```

**Why `uint16`:** 8,192 tokens fit into 2 bytes (max 65,535), so 97M tokens take ~195 MB instead of ~780 MB as int64. Training later reads the file with `np.memmap` without loading it into memory.

**Why a process pool:** encoding in pure Python is slow; documents are independent, so every core encodes its own.

**Tokenizer comparison:** the same validation text is encoded by my BPE, GPT-2 (`tiktoken` "gpt2") and GPT-4o (`o200k_base`), and tokens are divided by words:

| | vocabulary | tokens per Slovak word |
|---|---|---|
| my BPE | 8,192 | **2.36** |
| GPT-4o o200k | ~200,000 | 2.52 |
| GPT-2 | 50,257 | 3.39 |

**What it means:** on Slovak, GPT-2 needs 43 % more tokens, so 43 % more cost, latency and context. A small vocabulary trained on the target language beats a 24× larger multilingual one.

---

## Step 4 · Baselines: what must the GPT beat? ([`baselines.py`](baselines.py))

```python
uni = np.bincount(train, minlength=V) + K                      # unigram: token frequencies
pairs = np.bincount(train[:-1] * V + train[1:], minlength=V*V).reshape(V, V) + K   # bigram counts
bigram = -np.log(pairs[val[:-1], val[1:]]).mean()
```

- **Uniform:** every token 1/8192 → loss ln 8192 = 9.01.
- **Unigram:** how often each token appears, ignores context → perplexity 1,741.
- **Bigram:** P(next | previous token), the counting model from project 04 → perplexity **136.4**.

**Trick:** `train[:-1] * V + train[1:]` turns each pair (a, b) into one number a·V + b, so one `bincount` counts all 67M possible pairs at once.

**Why add-k smoothing (k = 0.01), fixed in advance:** a pair never seen in training would get probability 0 and loss ∞. k is fixed *before* looking at validation so the baseline is not tuned on it.

**Perplexity** = e^loss: "as unsure as picking uniformly from that many tokens". It only means something next to baselines on the **same** tokens.

---

## Step 5 · The model ([`model.py`](model.py))

```
token ids → token embedding + position embedding
          → N × [ x + Attention(LayerNorm(x)),  x + MLP(LayerNorm(x)) ]
          → LayerNorm → logits over 8,192 tokens
```

### 5a · Self-attention, written by hand
```python
q, k, v = self.qkv(x).split(C, dim=2)                                  # one matmul for all three
q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))   # (B, heads, T, 64)
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)                   # (B, h, T, T)
att = att.masked_fill(~self.mask[Tk - T : Tk, :Tk], float("-inf"))     # causal mask
att = F.softmax(att, dim=-1)
y = att @ v                                                             # weighted sum of values
y = y.transpose(1, 2).contiguous().view(B, T, C)                       # glue heads back
```

**How, in words:** every token makes three vectors:
- **query** q: "what am I looking for?"
- **key** k: "what do I contain?"
- **value** v: "what do I pass on if someone looks at me?"

The score q·k says how well a query matches each key. Softmax turns scores into weights that add up to 1, and the output is the weighted mix of values. This is where tokens exchange information.

**Why divide by √d:** the dot product of two 64-dimensional vectors grows with the dimension. Without scaling, softmax would become almost one-hot and the gradients would vanish.

**Causal mask:** before the softmax the scores for *future* tokens are set to −∞, so their weight is e^−∞ = 0. A token can only look back. Without it the model could just read the next token while training, and would learn nothing usable for generation.

**Multiple heads:** 384 dimensions are split into 6 heads of 64. Each head can learn a different kind of relationship (previous token, the subject of the sentence, …). The shapes are those from project 03.

**Why by hand and not `F.scaled_dot_product_attention`:** so the attention weights can be read out (`keep_att`) for the heatmaps in Step 8, and so the KV cache is visible in the code.

### 5b · The block
```python
x = x + self.att(self.ln1(x))     # tokens talk to each other
x = x + self.mlp(self.ln2(x))     # each token "thinks" on its own (384 → 1536 → 384, GELU)
```
- **Residual connection** (`x + …`): each block only adds a correction, so the gradient has a direct path through all layers (same idea as ResNet in project 11).
- **Pre-LayerNorm** (normalise *before* each sublayer, as in GPT-2) keeps the inputs of attention and MLP at a stable scale and makes training stable without careful tuning.

### 5c · Details in `GPT.__init__`
```python
self.head.weight = self.tok.weight      # weight tying
nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))   # for proj and mlp.2
```
- **Weight tying:** the output layer uses the same matrix as the input embedding. Reading a token and writing it are the same relation, and it saves 3.1M parameters (23 % of the base model).
- **Initialisation:** weights N(0, 0.02); the two layers per block that write into the residual stream are scaled down by 1/√(2·layers). Each block adds 2 contributions to x; without the scaling the sum would grow with depth at the start.
- **Learned positions:** attention itself does not know the order of tokens, so a position embedding (0…255) is added. Consequence: the model cannot go past 256 tokens.

Sizes: tiny (2 × 128), small (4 × 256), base (6 × 384, 13.8M parameters, 10.6M without embeddings).

---

## Step 6 · Training ([`train.py`](train.py))

### Batches
```python
ix = rng.integers(0, len(data) - context - 1, batch)
x = np.stack([data[i : i + context] for i in ix])
y = np.stack([data[i + 1 : i + 1 + context] for i in ix])   # the same window shifted by one
```
**Why shifted by one:** in one window of 256 tokens, position t predicts token t+1. Thanks to the causal mask that is 256 training examples per window at once, each seeing only its own past.

### Learning rate: warm-up and cosine
```python
if step < warmup: return peak * (step + 1) / warmup
return peak * (0.1 + 0.9 * 0.5 * (1 + cos(pi * progress)))
```
- **Warm-up (200 steps):** at the start, Adam's running averages of the gradient are still noise, so big steps could throw the model somewhere bad.
- **Cosine down to 10 %:** big steps in the middle, small at the end to settle (the same idea as one-cycle in project 11).
- Peak: 6e-4 for base, 1e-3 small, 2e-3 tiny: smaller models tolerate larger steps.

### AdamW with selective weight decay
```python
decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "tok" not in n and "pos" not in n]
torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                  lr=peak, betas=(0.9, 0.95))
```
**Why only matrices:** weight decay pulls weights towards zero (regularisation). For biases and LayerNorm gains that would only shift or shrink the signal; embeddings are left alone as in GPT-2. β₂ = 0.95 (instead of 0.999) lets Adam react faster to changing gradients, a common choice for transformers.

### The step
```python
_, loss = model(x, y)                                    # cross-entropy over all positions
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
opt.step()
```
**Gradient clipping:** if the total gradient norm exceeds 1.0, it is scaled down. One unusual batch cannot make a huge jump.

**Evaluation:** every 250 steps, 40 validation batches in `model.eval()` mode. Everything goes into `results_train_<size>.json`.

**Budget:** 5,000 steps × 64 × 256 = 82M tokens = 0.84 epochs, so the model never sees a training token twice. That is also why train loss and validation loss measure the same thing (README: error analysis).

**Result:** base **2.997 (perplexity 20.0)** in 41 minutes, against 136.4 for the bigram. Tiny 42.4, small 24.1.

---

## Step 7 · Sampling and the KV cache ([`model.py`](model.py) `generate`, `sample`)

### Sampling
```python
logits = logits / temperature                      # T < 1 sharper, T > 1 flatter
kth = torch.topk(logits, top_k).values[:, -1:]     # top-k: keep only the k best
p, order = probs.sort(descending=True)
drop = p.cumsum(-1) - p > top_p                    # top-p: the smallest set summing to p
return torch.multinomial(probs, 1)
```
- **Temperature** (project 08): T = 0.2 is almost greedy → repeating loops; T = 1.5 gives rare sub-word tokens enough probability to make broken words.
- **Top-k / top-p** cut the long tail of unlikely tokens instead of sharpening everything. `cumsum − p > top_p` keeps the token that crosses the threshold too.
- **Temperature 0** = greedy: always `argmax`.

### KV cache
```python
if "k" in cache:
    k = torch.cat([cache["k"], k], dim=2)
    v = torch.cat([cache["v"], v], dim=2)
cache["k"], cache["v"] = k, v
...
x = nxt if use_cache else idx      # with a cache, only the new token goes in
```
**Why it is possible:** thanks to the causal mask, earlier tokens never look at later ones, so their keys and values **never change** when a new token arrives. Without a cache the model pushes the whole sequence through again for every new token; with a cache, each layer remembers K and V and computes only the new token's q, k, v.

**The masking detail:** with a cache, T = 1 query and Tk keys. `self.mask[Tk - T : Tk, :Tk]` takes the row for the query's *absolute* position. Positions too: `start = caches[0]["k"].shape[2]` tells the model where it is.

**How it is checked** (`generate.py`): greedy generation with and without the cache gives **identical 253 tokens** on CPU and MPS. Speed: CPU 147 → 689 tokens/s (**4.7×**), MPS 173 → 172 (nothing). On the GPU with such a small model each step waits for kernel launches, not arithmetic, so removing arithmetic does not help. Measure before believing.

---

## Step 8 · Look inside: attention maps ([`generate.py`](generate.py) `attention`)

```python
model(torch.tensor([ids]), keep_att=True)
maps = [b.att.last_att[0].numpy() for b in model.blocks]   # per layer: (heads, T, T)
prev  = np.mean([a[i, i - 1] for i in range(1, T)])         # attention on the previous token
first = np.mean(a[1:, 0])                                   # attention parked on the first token
```

On the sentence *„Mária kúpila psa, ale ten pes ju nepočúval, lebo Mária ho nekŕmila.“*:
- **Layer 1, head 2** puts 51 % on the previous token: a diagonal. Such heads are building blocks of "induction" (if A was followed by B, predict B after A again).
- **Layer 3, head 4** puts 90 % on the first token: an **attention sink**. Softmax must hand out 100 %, so a head with nothing useful to do parks it on a harmless token.

**Why it matters:** it shows that the heads specialise without anyone telling them to, and that the numbers can be measured, not only looked at.

---

## Step 9 · Run everything

```bash
make all STEPS=5000      # data → tokenizer → baselines → train 3 sizes → generate → plot → lint
```

The README's numbers come from `--steps 5000`; the Makefile default is 6000, so pass `STEPS=5000` to reproduce them.

---

## Try it yourself

1. Train the BPE with `VOCAB_SIZE = 4096` and `16384`. How do tokens per word and the final perplexity change? (Compare loss per *character*, not per token, because tokens differ.)
2. Remove the causal mask in `Attention` and train tiny for 500 steps. Training loss will drop very low, so why is generation useless?
3. Generate with `temperature=0.7, top_p=0.9` instead of plain T = 0.2 on the prompt "V roku 1993". Does the loop disappear?

## Recap

| Step | Idea |
|---|---|
| 1 | Shuffle, then split by article; an end-of-text token between articles |
| 2 | BPE: bytes → merge the most frequent pair; incremental counts + heap = 43 s instead of hours |
| 3 | Tokenizer trained on Slovak: 2.36 tokens/word vs 3.39 for GPT-2 |
| 4 | Perplexity means something only against baselines on the same tokens (bigram 136) |
| 5 | Attention = softmax(q·k/√d) with a causal mask; residual + pre-LN; weight tying |
| 6 | Shifted windows; warm-up + cosine; AdamW on matrices only; gradient clipping |
| 7 | Temperature/top-k/top-p; KV cache works because old K, V never change — but measure the speed-up |
| 8 | Heads specialise: previous token, attention sink |
