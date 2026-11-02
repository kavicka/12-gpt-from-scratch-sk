# Tutoriál · 12 · GPT od nuly po slovensky

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál postaví projekt v poradí, v akom by si ho písal: dáta → tokenizer → súbory s tokenmi → baseliny → model → tréningový cyklus → vzorkovanie a KV cache → mapy attention. Každý krok vysvetľuje, **čo** kód robí, **prečo** je napísaný takto a **ako** funguje. README má všetky čísla.

**Cieľ:** byte-level BPE tokenizer a decoder-only transformer (GPT), oba napísané ručne, natrénované na slovenskej Wikipédii na Macu a zmerané oproti jednoduchým baselinom.

**Potrebuješ:** projekty 04 (makemore: predikcia ďalšieho znaku, tréningový cyklus), 08 (cross-entropy, softmax, teplota) a 11 (tréning v PyTorchi na MPS). Knižnice: PyTorch, `regex`, pyarrow, numpy; `tiktoken` len na porovnanie tokenizerov.

---

## Krok 1 · Získaj text ([`data.py`](data.py))

```python
MIN_CHARS = 500                    # preskoč jednoriadkové pahýle
texts = [t for t in texts if len(t) >= MIN_CHARS]
random.Random(0).shuffle(texts)    # zamiešaj celé články pred rozdelením
val, train = keep[:n_val], keep[n_val:]     # 1 % článkov na validáciu
(DATA / "sk_train.txt").write_text(EOT.join(train))   # EOT = "\n<|endoftext|>\n"
```

**Čo:** stiahne jeden 240 MB Parquet súbor slovenskej Wikipédie (dump 2023-11-01), ponechá články s aspoň 500 znakmi a rozdelí ich na 105 072 trénovacích a 1 061 validačných (297M / 2,8M znakov).

**Prečo vyhodiť krátke články:** väčšina z nich sú jednoriadkové pahýle obcí z tej istej šablóny („X je obec v okrese Y“). Tisíce kópií by model naučili opakovať túto šablónu.

**Prečo miešať celé články:** súbor je zoradený podľa názvu. Bez miešania by validácia bola jeden abecedný blok. Delenie po *článkoch*, nie po riadkoch, zároveň zabráni tomu, aby vety jedného článku skončili na oboch stranách.

**Prečo `<|endoftext|>`:** oddeľovač medzi článkami, z ktorého sa neskôr stane jeden špeciálny token. Model sa naučí, kde článok končí, a začiatok článku nie je prilepený ku koncu predošlého.

---

## Krok 2 · Napíš vlastný BPE tokenizer ([`bpe.py`](bpe.py))

Model pracuje s číslami, nie s písmenami. **Tokenizer** rozreže text na kúsky (tokeny) a každému dá id. Znaky by robili sekvencie veľmi dlhé; celé slová by potrebovali obrovský slovník a nevedeli by si poradiť s novými slovami. **BPE** (byte pair encoding) je niekde medzi: začne od bajtov a naučí sa časté kúsky.

### 2a · Predrozdelenie na kúsky (chunky)
```python
GPT2_SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```
Regex z GPT-2 rozdelí text na slovo s medzerou pred ním, číslo, interpunkciu alebo medzery. `\p{L}` znamená „akékoľvek písmeno v akomkoľvek jazyku“, takže `š`, `ô`, `ľ` zostanú vnútri slova. **Prečo:** zlúčenia nikdy neprekročia hranicu chunku, takže „mesto.“ sa nikdy nestane jedným tokenom aj s bodkou a slovník sa minie na skutočné kúsky slov.

### 2b · Začni od bajtov
```python
words = [list(c.encode("utf-8")) for c in chunk_freq]
```
Každý chunk začne ako svoje UTF-8 bajty, takže je 256 počiatočných symbolov a **nič nie je nikdy neznáme** (žiadny token `<unk>`). Písmená ako `š` zaberú 2 bajty; anglický tokenizer, ktorý sa ich nenaučil zlúčiť, zaplatí za jedno písmeno 2 tokeny. Preto je GPT-2 na slovenčine drahý.

### 2c · Tréning: zlúč najčastejšiu dvojicu, znova a znova
```python
neg, pair = heapq.heappop(heap)
if counts.get(pair, 0) != -neg: continue      # zastaraný záznam v halde, preskoč
new = 256 + len(merges); merges.append(pair)
for wi in where.pop(pair):                    # len chunky, ktoré túto dvojicu obsahujú
    for p in pairwise(w): counts[p] -= f      # odober staré dvojice
    ... nahraď (a, b) za `new` ...
    for p in pairwise(out): counts[p] += f    # pridaj nové dvojice
```
**Ako:** spočítaj každú susednú dvojicu symbolov (váženú tým, ako často sa chunk vyskytuje), zlúč najčastejšiu dvojicu do nového symbolu, opakuj 7 935-krát → 256 + 7 935 + 1 špeciálny = 8 192 tokenov.

**Prečo tak zložito:** naivná verzia po každom zlúčení prepočíta všetky dvojice vo všetkých chunkoch, čo by na 40M znakoch trvalo hodiny. Tu:
- `where[pair]` si pamätá, ktoré chunky dvojicu obsahujú, takže sa prepočítajú len tie;
- **halda (heap)** rýchlo vráti najčastejšiu dvojicu. Namiesto úpravy záznamov vnútri haldy sa vloží nový záznam a staré sa pri vybratí preskočia, ak ich počet už nesedí („lenivé mazanie“).

Výsledok: **43 s** na Macu. Na malom texte sa naučí presne tých istých 300 z 300 zlúčení ako naivná verzia, takže zrýchlenie nezmenilo výsledok.

**Prečo trénovať na 40M znakoch a nie na všetkých 297M:** frekvencie dvojíc sa ustália oveľa skôr; viac textu len stojí čas.

### 2d · Kódovanie
```python
pair = min(pairwise(ids), key=lambda p: self.ranks.get(p, 1 << 30))
```
Pri kódovaní nového chunku aplikuj zlúčenie, ktoré sa **naučilo najskôr** spomedzi prítomných dvojíc, a opakuj, kým žiadna dvojica nie je v zozname zlúčení. **Prečo v tomto poradí:** v tomto poradí tokeny vznikali pri tréningu, takže kódovanie ich zreprodukuje. Výsledky sa cachujú pre každý chunk, lebo Wikipédia opakuje tie isté slová miliónkrát.

---

## Krok 3 · Zakóduj všetko a porovnaj tokenizery ([`prepare.py`](prepare.py))

```python
parts = pool.map(_encode, docs, chunksize=64)                       # všetky jadrá CPU
ids = np.fromiter((i for p in parts for i in (*p, bpe.eot)), dtype=np.uint16)
ids.tofile(DATA / "train.bin")
```

**Prečo `uint16`:** 8 192 tokenov sa zmestí do 2 bajtov (max 65 535), takže 97M tokenov zaberie ~195 MB namiesto ~780 MB ako int64. Tréning potom číta súbor cez `np.memmap` bez načítania do pamäte.

**Prečo pool procesov:** kódovanie v čistom Pythone je pomalé; dokumenty sú nezávislé, takže každé jadro kóduje svoje.

**Porovnanie tokenizerov:** ten istý validačný text zakóduje môj BPE, GPT-2 (`tiktoken` „gpt2“) a GPT-4o (`o200k_base`) a počet tokenov sa vydelí počtom slov:

| | slovník | tokenov na slovenské slovo |
|---|---|---|
| môj BPE | 8 192 | **2,36** |
| GPT-4o o200k | ~200 000 | 2,52 |
| GPT-2 | 50 257 | 3,39 |

**Čo to znamená:** na slovenčine potrebuje GPT-2 o 43 % viac tokenov, teda o 43 % viac peňazí, latencie a kontextu. Malý slovník natrénovaný na cieľovom jazyku porazí 24× väčší viacjazyčný.

---

## Krok 4 · Baseliny: čo musí GPT poraziť? ([`baselines.py`](baselines.py))

```python
uni = np.bincount(train, minlength=V) + K                      # unigram: frekvencie tokenov
pairs = np.bincount(train[:-1] * V + train[1:], minlength=V*V).reshape(V, V) + K   # počty bigramov
bigram = -np.log(pairs[val[:-1], val[1:]]).mean()
```

- **Rovnomerný:** každý token 1/8192 → loss ln 8192 = 9,01.
- **Unigram:** ako často sa každý token vyskytuje, kontext ignoruje → perplexita 1 741.
- **Bigram:** P(ďalší | predošlý token), počítací model z projektu 04 → perplexita **136,4**.

**Trik:** `train[:-1] * V + train[1:]` premení každú dvojicu (a, b) na jedno číslo a·V + b, takže jeden `bincount` spočíta všetkých 67M možných dvojíc naraz.

**Prečo add-k vyhladzovanie (k = 0,01), pevne vopred:** dvojica, ktorú tréning nevidel, by dostala pravdepodobnosť 0 a loss ∞. k je zvolené *pred* pohľadom na validáciu, aby baseline nebol na nej vyladený.

**Perplexita** = e^loss: „taká neistota, ako keby si vyberal rovnomerne z toľkých tokenov“. Zmysel má len vedľa baselinov na **tých istých** tokenoch.

---

## Krok 5 · Model ([`model.py`](model.py))

```
id tokenov → embedding tokenu + embedding pozície
           → N × [ x + Attention(LayerNorm(x)),  x + MLP(LayerNorm(x)) ]
           → LayerNorm → logity cez 8 192 tokenov
```

### 5a · Self-attention, napísaná ručne
```python
q, k, v = self.qkv(x).split(C, dim=2)                                  # jedno násobenie pre všetky tri
q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))   # (B, hlavy, T, 64)
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)                   # (B, h, T, T)
att = att.masked_fill(~self.mask[Tk - T : Tk, :Tk], float("-inf"))     # kauzálna maska
att = F.softmax(att, dim=-1)
y = att @ v                                                             # vážený súčet hodnôt
y = y.transpose(1, 2).contiguous().view(B, T, C)                       # zlep hlavy späť
```

**Ako, slovami:** každý token vyrobí tri vektory:
- **query** q: „čo hľadám?“
- **key** k: „čo obsahujem?“
- **value** v: „čo odovzdám, keď sa na mňa niekto pozrie?“

Skóre q·k hovorí, ako dobre query sedí ku každému key. Softmax z nich spraví váhy so súčtom 1 a výstup je vážená zmes hodnôt. Tu si tokeny vymieňajú informácie.

**Prečo deliť √d:** skalárny súčin dvoch 64-rozmerných vektorov rastie s dimenziou. Bez škálovania by softmax bol takmer one-hot a gradienty by zmizli.

**Kauzálna maska:** pred softmaxom sa skóre pre *budúce* tokeny nastavia na −∞, takže ich váha je e^−∞ = 0. Token sa môže pozerať len dozadu. Bez nej by model pri tréningu jednoducho prečítal nasledujúci token a nenaučil by sa nič použiteľné na generovanie.

**Viac hláv:** 384 dimenzií sa rozdelí na 6 hláv po 64. Každá hlava sa môže naučiť iný druh vzťahu (predošlý token, podmet vety, …). Tvary sú tie z projektu 03.

**Prečo ručne a nie `F.scaled_dot_product_attention`:** aby sa dali váhy attention prečítať (`keep_att`) pre heatmapy v Kroku 8 a aby bola KV cache viditeľná v kóde.

### 5b · Blok
```python
x = x + self.att(self.ln1(x))     # tokeny sa rozprávajú medzi sebou
x = x + self.mlp(self.ln2(x))     # každý token „premýšľa“ sám (384 → 1536 → 384, GELU)
```
- **Reziduálne spojenie** (`x + …`): každý blok len pridá korekciu, takže gradient má priamu cestu cez všetky vrstvy (rovnaká myšlienka ako ResNet v projekte 11).
- **Pre-LayerNorm** (normalizuj *pred* každou podvrstvou, ako v GPT-2) drží vstupy attention a MLP v stabilnej mierke a tréning je stabilný bez opatrného ladenia.

### 5c · Detaily v `GPT.__init__`
```python
self.head.weight = self.tok.weight      # weight tying
nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))   # pre proj a mlp.2
```
- **Weight tying:** výstupná vrstva používa tú istú maticu ako vstupný embedding. Čítanie tokenu a jeho písanie je ten istý vzťah a ušetrí to 3,1M parametrov (23 % základného modelu).
- **Inicializácia:** váhy N(0; 0,02); dve vrstvy v každom bloku, ktoré zapisujú do reziduálneho prúdu, sa zmenšia o 1/√(2·vrstvy). Každý blok pridá k x dva príspevky; bez škálovania by súčet na začiatku rástol s hĺbkou.
- **Naučené pozície:** samotná attention nepozná poradie tokenov, preto sa pripočíta embedding pozície (0…255). Dôsledok: model nevie ísť za 256 tokenov.

Veľkosti: tiny (2 × 128), small (4 × 256), base (6 × 384, 13,8M parametrov, 10,6M bez embeddingov).

---

## Krok 6 · Tréning ([`train.py`](train.py))

### Batche
```python
ix = rng.integers(0, len(data) - context - 1, batch)
x = np.stack([data[i : i + context] for i in ix])
y = np.stack([data[i + 1 : i + 1 + context] for i in ix])   # to isté okno posunuté o jedna
```
**Prečo posunuté o jedna:** v jednom okne 256 tokenov pozícia t predpovedá token t+1. Vďaka kauzálnej maske je to naraz 256 tréningových príkladov na okno, každý vidí len svoju minulosť.

### Learning rate: warm-up a kosínus
```python
if step < warmup: return peak * (step + 1) / warmup
return peak * (0.1 + 0.9 * 0.5 * (1 + cos(pi * progress)))
```
- **Warm-up (200 krokov):** na začiatku sú Adamove kĺzavé priemery gradientu ešte šum, takže veľké kroky by mohli model odhodiť niekam zle.
- **Kosínus až na 10 %:** veľké kroky v strede, malé na konci, aby sa usadil (rovnaká myšlienka ako one-cycle v projekte 11).
- Maximum: 6e-4 pre base, 1e-3 small, 2e-3 tiny: menšie modely znesú väčšie kroky.

### AdamW s výberovým weight decay
```python
decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "tok" not in n and "pos" not in n]
torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                  lr=peak, betas=(0.9, 0.95))
```
**Prečo len matice:** weight decay ťahá váhy k nule (regularizácia). Pri biasoch a ziskoch LayerNormu by to len posúvalo alebo zmenšovalo signál; embeddingy sa nechávajú ako v GPT-2. β₂ = 0,95 (namiesto 0,999) dovolí Adamu rýchlejšie reagovať na meniace sa gradienty, bežná voľba pri transformeroch.

### Krok
```python
_, loss = model(x, y)                                    # cross-entropy cez všetky pozície
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # orezanie gradientu
opt.step()
```
**Orezanie gradientu (clipping):** ak celková norma gradientu prekročí 1,0, zmenší sa. Jeden nezvyčajný batch nespraví obrovský skok.

**Vyhodnotenie:** každých 250 krokov 40 validačných batchov v režime `model.eval()`. Všetko ide do `results_train_<size>.json`.

**Rozpočet:** 5 000 krokov × 64 × 256 = 82M tokenov = 0,84 epochy, takže model nikdy nevidí trénovací token dvakrát. Preto aj train loss a validačná loss merajú to isté (README: analýza chýb).

**Výsledok:** base **2,997 (perplexita 20,0)** za 41 minút oproti 136,4 pre bigram. Tiny 42,4, small 24,1.

---

## Krok 7 · Vzorkovanie a KV cache ([`model.py`](model.py) `generate`, `sample`)

### Vzorkovanie
```python
logits = logits / temperature                      # T < 1 ostrejšie, T > 1 plochejšie
kth = torch.topk(logits, top_k).values[:, -1:]     # top-k: nechaj len k najlepších
p, order = probs.sort(descending=True)
drop = p.cumsum(-1) - p > top_p                    # top-p: najmenšia množina so súčtom p
return torch.multinomial(probs, 1)
```
- **Teplota** (projekt 08): T = 0,2 je takmer greedy → opakujúce sa slučky; T = 1,5 dá zriedkavým kúskom slov dosť pravdepodobnosti na rozbité slová.
- **Top-k / top-p** odrežú dlhý chvost nepravdepodobných tokenov namiesto zostrenia všetkého. `cumsum − p > top_p` ponechá aj token, ktorý prah prekročí.
- **Teplota 0** = greedy: vždy `argmax`.

### KV cache
```python
if "k" in cache:
    k = torch.cat([cache["k"], k], dim=2)
    v = torch.cat([cache["v"], v], dim=2)
cache["k"], cache["v"] = k, v
...
x = nxt if use_cache else idx      # s cache ide dnu len nový token
```
**Prečo je to možné:** vďaka kauzálnej maske sa skoršie tokeny nikdy nepozerajú na neskoršie, takže ich keys a values sa **nikdy nezmenia**, keď príde nový token. Bez cache model pre každý nový token pošle celú sekvenciu znova; s cache si každá vrstva pamätá K a V a počíta len q, k, v nového tokenu.

**Detail masky:** s cache je T = 1 query a Tk keys. `self.mask[Tk - T : Tk, :Tk]` vezme riadok pre *absolútnu* pozíciu query. Pozície tiež: `start = caches[0]["k"].shape[2]` povie modelu, kde je.

**Ako sa to overí** (`generate.py`): greedy generovanie s cache aj bez nej dá **identických 253 tokenov** na CPU aj MPS. Rýchlosť: CPU 147 → 689 tokenov/s (**4,7×**), MPS 173 → 172 (nič). Na GPU s takým malým modelom každý krok čaká na spúšťanie kernelov, nie na aritmetiku, takže odobrať aritmetiku nepomôže. Najprv zmeraj, potom ver.

---

## Krok 8 · Pozri sa dnu: mapy attention ([`generate.py`](generate.py) `attention`)

```python
model(torch.tensor([ids]), keep_att=True)
maps = [b.att.last_att[0].numpy() for b in model.blocks]   # pre vrstvu: (hlavy, T, T)
prev  = np.mean([a[i, i - 1] for i in range(1, T)])         # attention na predošlý token
first = np.mean(a[1:, 0])                                   # attention zaparkovaná na prvom tokene
```

Na vete *„Mária kúpila psa, ale ten pes ju nepočúval, lebo Mária ho nekŕmila.“*:
- **Vrstva 1, hlava 2** dáva 51 % na predošlý token: diagonála. Takéto hlavy sú stavebné kamene „indukcie“ (ak po A nasledovalo B, predpovedz B po A znova).
- **Vrstva 3, hlava 4** dáva 90 % na prvý token: **attention sink**. Softmax musí rozdať 100 %, takže hlava, ktorá nemá čo užitočné robiť, to zaparkuje na neškodnom tokene.

**Prečo na tom záleží:** ukazuje to, že hlavy sa špecializujú bez toho, aby im to niekto povedal, a že sa to dá zmerať, nie len pozerať.

---

## Krok 9 · Spusti všetko

```bash
make all STEPS=5000      # dáta → tokenizer → baseliny → tréning 3 veľkostí → generovanie → grafy → lint
```

Čísla v README sú z `--steps 5000`; Makefile má predvolene 6000, takže na zopakovanie zadaj `STEPS=5000`.

---

## Vyskúšaj sám

1. Natrénuj BPE s `VOCAB_SIZE = 4096` a `16384`. Ako sa zmení počet tokenov na slovo a výsledná perplexita? (Porovnávaj loss na *znak*, nie na token, lebo tokeny sú rôzne.)
2. Odstráň kauzálnu masku v `Attention` a natrénuj tiny na 500 krokov. Trénovacia loss klesne veľmi nízko, prečo je teda generovanie nepoužiteľné?
3. Generuj s `temperature=0.7, top_p=0.9` namiesto čistého T = 0,2 na prompte „V roku 1993“. Zmizne slučka?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1 | Zamiešaj, potom deľ po článkoch; medzi článkami token konca textu |
| 2 | BPE: bajty → zlúč najčastejšiu dvojicu; inkrementálne počty + halda = 43 s namiesto hodín |
| 3 | Tokenizer natrénovaný na slovenčine: 2,36 tokenu/slovo vs 3,39 pre GPT-2 |
| 4 | Perplexita má zmysel len oproti baselinom na tých istých tokenoch (bigram 136) |
| 5 | Attention = softmax(q·k/√d) s kauzálnou maskou; reziduál + pre-LN; weight tying |
| 6 | Posunuté okná; warm-up + kosínus; AdamW len na matice; orezanie gradientu |
| 7 | Teplota/top-k/top-p; KV cache funguje, lebo staré K, V sa nemenia, ale zrýchlenie zmeraj |
| 8 | Hlavy sa špecializujú: predošlý token, attention sink |
