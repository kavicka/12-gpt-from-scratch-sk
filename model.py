"""A decoder-only transformer (GPT), written out layer by layer.

token ids → token embedding + position embedding → N × Block → LayerNorm → logits over the vocabulary
Block = x + Attention(LayerNorm(x)), then x + MLP(LayerNorm(x))   (pre-norm residual, as in GPT-2)

The attention is written by hand (no F.scaled_dot_product_attention) so that the attention weights
can be read out for the heatmaps, and so that the KV cache is visible in the code.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    context: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


SIZES = {
    "tiny": Config(d_model=128, n_layers=2, n_heads=4),
    "small": Config(d_model=256, n_layers=4, n_heads=4),
    "base": Config(d_model=384, n_layers=6, n_heads=6),
}


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)  # Q, K and V for all heads in one matmul
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.context, cfg.context, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
        self.last_att: torch.Tensor | None = None  # kept only when asked, for the heatmaps

    def forward(self, x: torch.Tensor, cache: dict | None = None, keep_att: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, heads, T, head_dim): each head works in its own smaller subspace
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            # KV cache: keys and values of earlier tokens never change (causal mask), so keep them and
            # compute only the new token's q, k, v. One step costs O(T) instead of O(T²).
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        Tk = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B, h, T, Tk): how well each query matches each key
        # causal mask: query at absolute position Tk−T+i may only look at keys 0 … Tk−T+i
        att = att.masked_fill(~self.mask[Tk - T : Tk, :Tk], float("-inf"))
        att = F.softmax(att, dim=-1)
        if keep_att:
            self.last_att = att.detach()
        y = self.drop(att) @ v  # weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # glue the heads back together
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.att = Attention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, keep_att=False):
        x = x + self.att(self.ln1(x), cache, keep_att)  # tokens talk to each other
        return x + self.mlp(self.ln2(x))  # each token thinks on its own


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)  # learned positions (GPT-2 style)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying: the same matrix reads and writes tokens
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "mlp.2.weight")):
                # residual branches add up over layers; scale them down so the sum stays O(1) at init
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.pos.weight.numel() - (self.tok.weight.numel() if non_embedding else 0)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, caches: list | None = None, keep_att: bool = False
    ):
        T = idx.shape[1]
        start = caches[0]["k"].shape[2] if caches and "k" in caches[0] else 0
        pos = torch.arange(start, start + T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None, keep_att)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
        generator: torch.Generator | None = None,
    ):
        """Sample new tokens one at a time. temperature=0 means greedy (always the most likely token)."""
        caches = [{} for _ in self.blocks] if use_cache else None
        x = idx
        for _ in range(new_tokens):
            if idx.shape[1] >= self.cfg.context:
                break  # learned positions exist only up to `context`
            logits, _ = self(x, caches=caches)
            logits = logits[:, -1, :]
            nxt = sample(logits, temperature, top_k, top_p, generator)
            idx = torch.cat([idx, nxt], dim=1)
            x = nxt if use_cache else idx  # with a cache only the new token goes in; without, everything again
        return idx


def sample(logits, temperature=1.0, top_k=None, top_p=None, generator=None) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature  # T < 1 sharpens the distribution, T > 1 flattens it
    if top_k is not None:
        kth = torch.topk(logits, top_k).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p is not None:  # nucleus: keep the smallest set of tokens whose probability adds up to top_p
        p, order = probs.sort(descending=True)
        drop = p.cumsum(-1) - p > top_p
        p = p.masked_fill(drop, 0)
        probs = torch.zeros_like(probs).scatter(-1, order, p)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator)
