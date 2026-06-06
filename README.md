# living-kv-cache

**Recall a fact across 800,000 tokens of context on a consumer 8 GB GPU.**

A small research PoC: keep only a small *hot window* of the KV-cache in GPU VRAM, stream the rest ("cold"
KV) to cheap CPU RAM, and on a query use a tiny in-VRAM catalog to pull back only the relevant piece.
VRAM stays **small and constant** no matter how long the context gets — the limit moves from VRAM to RAM.

> ⚠️ **Honest framing first.** The **core mechanism is not new**: recalling evicted KV from CPU memory was
> done by **ArkVale (NeurIPS 2024)** — async-copy KV pages to CPU, summarize them into a digest, and recall
> the important ones — and by **RetrievalAttention (2024)** via CPU + ANN retrieval. This repo does **not**
> claim that idea. What is different here is narrow but real: (1) the digest is a **semantic self-embedding
> taken from the model's own hidden states** (de-anisotropized), instead of ArkVale's bounding-volume-of-keys
> or RetrievalAttention's trained ANN index; (2) a **reproducible demonstration on a cheap 8 GB consumer GPU
> up to 800k tokens**; (3) **honestly reported failure modes**. It is a PoC, not a production library, and the
> overlap with prior work is large — please read the prior-art list below.

---

> **New (Jun 2026): [`llamacpp/`](llamacpp/) — the same mechanics ported onto stock llama.cpp** (no kernel
> fork, public C API only): bit-exact KV cut/restore, streaming 131k tokens through a 32×-smaller buffer,
> w-dynamics (decay/recharge/eviction/revival), q8_0 cold store, constant ~84 tok/s decode where the full
> cache OOMs. See [llamacpp/README.md](llamacpp/README.md).

## The idea in plain words

A transformer writes a "card" (a key/value vector) for every token it reads, and to produce the next token
it looks at **all** the cards. Normally those cards live on a tiny, fast shelf — GPU video memory. On an 8 GB
card that shelf fills up around ~110k tokens, and then the model either can't continue or starts to break.

This repo does it differently:

- **Hot window** (last `WIN` tokens) stays on the fast shelf (VRAM).
- **Everything else** goes into a big cheap drawer (CPU RAM), optionally 8-bit compressed.
- A small **catalog** (one semantic embedding per sentence, kept in VRAM) finds the relevant cards on a query
  and pulls back only a small **budget** of them.

So the fast shelf stays small and constant — and the information isn't lost, just kept in a cheaper place.

Positions stay correct by construction: keys are stored already rotary-rotated at their absolute position,
and the query is fed at its true absolute position, so relative distances are exact even for far-apart tokens.

## What we measured

Model: **Qwen2.5-7B-Instruct-1M** in 4-bit, on a single **RTX 5060 (8 GB)**. Single early needle, recalled
after a long run of distractors. `rank 0` = the fact is the top prediction.

| context length | fp16 cold store | 8-bit cold store | peak VRAM | cold store (RAM) |
|---:|:---:|:---:|:---:|:---:|
| 120k | rank 0, p 0.98 | — | ~6.1 GB | ~7 GB |
| 200k | rank 0, p 0.96 | rank 0, p 0.96 | ~6.1 GB | ~6 / 12 GB |
| 400k | rank 0, p 0.74 | — | ~6.1 GB | ~22 GB |
| 500k | (RAM-limited) | rank 0, p 0.07 | ~6.1 GB | ~15 GB |
| **800k** | — | **rank 1, p 0.21** | **~6.1 GB** | ~23 GB |

**The fact stays at rank ≤ 1 all the way to 800,000 tokens, with constant ~6 GB VRAM.**

A 1M-trained model is required (see caveats). For comparison, a 128k-context model (MiniCPM5-1B) collapses to
garbage by ~120k **no matter what we do** — that is the model's own competence limit, which this method does
not change.

## Honest caveats (please read before judging)

1. **The model must be trained for long context.** We used Qwen2.5-7B-**1M**. A short-context model breaks
   past its *effective* range (~50–70 % of nominal, the RULER effect) regardless of this machinery.
2. **Confidence decays with length.** The fact stays top-1/top-2, but the model grows less certain
   (p 0.98 at 120k → p 0.21 at 800k).
3. **Single synthetic needle.** Many *same-type* needles (e.g. 8 "the code for CITY is …" lines) defeat the
   tiny-budget probe — full attention is needed there. This is a real limitation of the *self-embedding*
   probe in this python PoC. *(Update: in the [llama.cpp port](llamacpp/) an **external text-embedding
   catalog** finds 8/8 same-type needles — entity names are strong lexical anchors there; restoring more
   than the single best match can still distract generation, so selection should be confidence-aware.)*
4. **Windowed prefill.** Late tokens attend only to a `WIN`-token window, so this demonstrates *retrieval of
   a stored fact*, not full bidirectional understanding of the whole context.
5. **4-bit cold store is NOT lossless** (it degrades recall even at 32k). Use `BITS=8` (lossless recall, ~2×
   RAM saving) or `BITS=16`. True 4× compression needs a real TurboQuant (Lloyd-Max) quantizer + bit-packing,
   which is not implemented.
6. **The PoC is not speed-optimised** for huge prefills. It is a correctness/recall demo.

## Which models this applies to

Needs a **standard growing KV-cache** (ordinary softmax attention). Open-weight, hookable via HF transformers.

- ✅ **Directly:** Llama-3.x / Llama-4, Qwen2.5 (incl. 7B/14B/72B-1M), Mistral / Mixtral, GLM-4, Command-R+,
  Yi, gpt-oss. (MoE is fine — only the FFN is sparse, attention is still full.)
- ⚙️ **With an adapter (great fit):** DeepSeek-V3/V4, Kimi — they use **MLA** (latent KV). The cold store
  becomes the latent KV, which is already much smaller → reaches even further.
- ❌ **Does not apply:** Mamba/SSM and linear-attention hybrids (Qwen3.5 GatedDeltaNet, Granite-4, RWKV,
  Falcon-Mamba, LFM) — they compress history into a fixed state, so there is no growing KV to spill.

We only *tested* MiniCPM5-1B and Qwen2.5-7B-1M. The rest is architecture-level applicability, untested.

## Install & run

```bash
pip install -r requirements.txt           # torch, transformers, bitsandbytes, accelerate
# 8 GB GPU, ~200k context, 8-bit cold store:
MODEL_DIR=Qwen/Qwen2.5-7B-Instruct-1M N=200000 BITS=8 python demo.py
```

The first run downloads the model (~15 GB). `N` = context length, `BITS` ∈ {16, 8, 4}, `WIN` = hot window,
`BUDGET` = cards pulled per query.

## API

```python
from living_kv import LivingKVRunner
runner = LivingKVRunner(model, tokenizer, win=4096, budget=256, lmid=18, bits=8)
runner.prefill(token_ids, segments)          # segments = list of (start, end) token spans (e.g. sentences)
rank, prob, seg = runner.answer(question, answer_token)
```

## License

MIT. See [LICENSE](LICENSE).

## Prior art (verified against the sources, Jun 2026)

The closest precedents — read them first, this repo is a small variation on them:

- **ArkVale** — *Efficient Generative LLM Inference with Recallable Key-Value Eviction*, NeurIPS 2024.
  Async-copies KV pages to CPU, summarizes each page into a digest (bounding-volume of the keys), and
  recalls pages that regain importance. Code: https://github.com/pku-liang/ArkVale. **This is the direct
  precedent for the core idea.**
- **RetrievalAttention** — arXiv [2409.10516](https://arxiv.org/abs/2409.10516). KV in CPU, attention-aware
  ANN retrieval; ~1–3 % of KV accessed ≈ full attention. Demonstrated on a 24 GB RTX 4090 at 128k.

Related but a different problem (do **not** mistake these for the same thing — earlier drafts of this README
mis-cited them; corrected after checking the sources):

- **RetentiveKV** — arXiv [2605.04075](https://arxiv.org/abs/2605.04075). *Multimodal* KV eviction
  (deferred importance of visual tokens), entropy-driven; has a reactivation idea but a different domain.
- **FadeMem** — arXiv [2601.18642](https://arxiv.org/abs/2601.18642). Biologically-inspired forgetting for
  **agent memory** (a memory store for LLM agents), **not** the KV cache.

Baselines / components this builds on:

- **StreamingLLM**, **H2O** — the windowing / heavy-hitter eviction baselines.
- **TurboQuant** (Zandieh et al., ICLR 2026) — WHT + Lloyd-Max KV quantization (we use a uniform WHT+8-bit
  approximation of it for the cold store; the real Lloyd-Max version would compress further).
