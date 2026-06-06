# kvmem — living-KV session memory (ingest once, ask forever)

Feed text into a model **once**; ask questions in a **later, separate process**. The model
recalls from its own stored KV-cache — no re-reading the source, no RAG re-prefill. Built on the
[llama.cpp port](../llamacpp) of living-KV (streaming spill + nomic catalog + w-dynamics + q8_0).

```bash
kvmem ingest mystore notes.txt          # stream text through a small KV buffer, spill to disk
kvmem ingest mystore more.txt           # append (positions continue across files)
kvmem ask    mystore "what's the gate code?"   # fresh process: probe -> restore relevant KV -> answer
kvmem chat   mystore                     # interactive REPL; every turn is committed to the store,
                                         # so a LATER chat/ask process remembers this conversation
kvmem defrag mystore                     # rebuild the store from its stored text; set MODEL=new.gguf
                                         # to migrate a store to a different model
kvmem stats  mystore                     # manifest + hottest segments by weight
```

## How it differs from RAG

RAG hands the model **text to re-read**. kvmem hands the model **its own state from when it read
the text** — recall, not re-reading. Restore is milliseconds; no re-prefill of the source.

## What's verified (Qwen2.5-7B-1M Q4, RTX 5060 8 GB)

- **Cross-process recall**: ingest a 22k-token doc with planted facts; a *fresh* `ask` process
  answers correctly with no source re-prefill (`gate code → 7341`, `telescope key → blue flowerpot`).
- **Append**: a second `ingest` continues positions; `ask` recalls facts from either file.
- **Living weights persist**: each `ask` decays all segment weights and recharges the ones it used,
  written back to the catalog — so `stats` surfaces the most-queried facts at the top across runs.
- **Model-locked store**: `ask` refuses a store built by a different model/quant (KV is not portable)
  with a clear message instead of producing garbage.
- **Low-confidence honesty**: a question whose answer isn't in the store is flagged
  `⚠ LOW CONFIDENCE` (top probe score below threshold) and exits non-zero.

**Copy-bias (M2):** exact verbatim recall of a *novel multi-token string* (e.g. an invented
hostname `cobalt-finch`) used to get paraphrased even though the correct segment was retrieved.
Generation is greedy, so this is not a sampling issue. Two fixes landed together: a catalog
parsing bug that silently drifted the stored text away from the KV (the likely root cause), and a
**copy bias** — tokens that occur in the recalled segments' *stored text* get a logit bonus
(`COPYB=2.0`). The KV path stays pure: the stored text contributes token IDs only, it is never
re-prefilled. Verified on Qwen2.5-7B-1M: all three planted needles, including `cobalt-finch`,
now come back verbatim; honestly, the COPYB=0 control also passed there, so on a capable model
the parsing fix may carry most of the weight — the bias measurably helps small models stop
cleanly instead of hallucinating a fake Q&A continuation.

**Verified (M2, Qwen2.5-7B-1M Q4):** model-migration round-trip — a store built by Qwen was
migrated to TinyLlama (different tokenizer, 25.3k → 31.6k tokens) and back, after which the exact
answer (`7341`) returned; the probe finds the same segment throughout (embeddings depend only on
the text). Cross-session chat memory: a fact told in one `chat` process is recalled verbatim by a
fresh one.

**chat doc-recall (fixed via ask-fallback):** answering an *ingested-document* fact from inside
`chat` used to fail once the conversation had a prior turn — a clean A/B (Qwen2.5-7B-1M) showed
the cause was *not* distance from the live head (output was bit-identical with `llama_memory_seq_add`
re-positioning on or off) but the **conversational prior** overriding the injected KV: with a prior
turn present the model trusts the dialogue framing and says the fact "wasn't mentioned". The fix
(`FALLBACK=1`, default): on a confident probe hit, answer in a scratch sequence ask-style
(BOS + segment + question, no dialogue tail), then teacher-force the result back into the
conversation so continuity and the store stay intact. Verified: with the fix the doc fact is
recalled verbatim; with `FALLBACK=0` the old "wasn't mentioned" failure reproduces.

## Hybrid probe + extractive answers (logs)

The probe fuses two channels by RRF: **semantic** (nomic cosine) and **lexical** (IDF-weighted
exact-term overlap over the stored text). Pure semantic ranks *topical* neighbours for rare exact
terms; the lexical channel lifts the segment that literally contains the term. Verified on a store
of real chat logs: the lexical channel surfaced the right segment for an exact term (`kvmem`) that
pure cosine had ranked 0.66/topical.

`EXTRACT=1 kvmem ask <store> "q"` returns the **stored text** of the recalled segments instead of
generating. This is the honest mode for messy logs: free generation confabulates over real terms
that collide with the model's prior (e.g. `kvmem` → "virtual memory management", `touch
calibration` → "xinput_calibrator") *even when the correct segment is retrieved and adjacent* —
prior-dominance, not a retrieval miss (invented needles like `cobalt-finch` have no competing
prior and generate verbatim). EXTRACT sidesteps it by quoting what was actually stored.

Low-confidence is judged on the **semantic** channel only (`sem < THRESH`): semantic cosine is
the "is this topic in memory at all" signal; an incidental rare-word match must not pass an absent
fact as confident.

**Use it as:** a **fact store** answer well with generation (planted facts, notes, configs); a
**log store** is reliable with the hybrid probe + `EXTRACT=1` (grounded retrieval), while free
generation over arbitrary logs remains confabulation-prone.

## Grounded generation (M6: `GROUND=1`)

`GROUND=1 kvmem ask <store> "q"` answers **in the model's own words** without the confabulation:
the probe-picked **stored text goes into the prompt** as evidence (chat-template, "use only the
excerpts"), instead of relying on far-away KV. Prior-dominance loses to text under the model's
nose — that is the regime instruct models are trained for. Verified on a 38k-segment store of
real session logs: `what is kvmem` returns the correct tool description (the same question used
to produce "virtualized memory"); a related-but-partial topic gets an honest "here is what the
notes say, the decision itself isn't recorded"; an absent fact refuses with exit 2. Uses the GPU
(the LLM must load); `EXTRACT` stays GPU-free.

Two probe details only the text modes use:
- **Header expansion** (`MINCH=300`): tiny picked segments (markdown headers — "## what kvmem
  can do") are pointers to the body right after them; each pick is extended with document-order
  neighbours until ~MINCH chars. Caught on the full log store, where headers beat bodies.
- **No GAP cut**: the confidence gap (poc18f) protects KV-restore generation from
  similar-but-wrong KV; for prompt reading an extra excerpt is cheap and brings the defining
  segment along. EXTRACT/GROUND always read the full `SEL`.

## Embedder per store (M6: `EMB`, `reembed`)

The embedding model is a **store property** (manifest: `emb_model`, `emb_dim`); mixing embedders
in one `embs.f32` is garbage, so `ingest` into a store embedded with a different model fails
loudly. New stores: `EMB=bge-m3 kvmem ingest ...` (default stays `nomic-embed-text`; legacy
stores without the manifest keys load as nomic/768). Migration:

```bash
EMB=bge-m3 kvmem reembed mystore     # rewrites embs.f32 from the stored text (~10 seg/s via ollama)
```

Why bge-m3: nomic is EN-centric — on Russian queries it returns wrong segments with *high*
cosine (anisotropy; measured 0.80 on an unrelated segment). bge-m3 fixed the RU channel on the
real log store ("что мы решили про вайфай планшета" → the exact wifi complaint, where nomic
returned an unrelated discussion). **Cosine scales differ per embedder** — recalibrate `THRESH`
after a reembed (bge-m3 on the log store: positives 0.61–0.75, absent 0.47–0.53 → `THRESH=0.57`;
the 0.65 default is nomic-scale). On tiny stores positives sit lower (~0.52–0.59) — the
low-confidence flag is a warning, tune per store.

## Store layout (`mystore/`)

```
manifest.txt     model_file, model_bytes, kv_type, build, total_tokens, n_segments, emb_model, emb_dim
catalog.tsv      id  lo  hi  w  text          (source text kept next to KV = re-decode insurance)
embs.f32         float32[n_segments][emb_dim]  (probe embeddings; emb_model in the manifest)
segments/N.kv    serialized KV range for segment N (llama_state_seq; build-sensitive; absent in TEXTONLY stores)
```

The source text is stored beside the KV on purpose: if the model, quant, or llama.cpp build
changes, the KV is invalid but the text lets you re-ingest. The manifest records all three so a
mismatched `ask` fails loudly.

## Build

Built by the repo installer (`../llamacpp/install.sh`) or manually:

```bash
g++ -O2 -o kvmem kvmem.cpp -I <llama.cpp>/include -I <llama.cpp>/ggml/include \
    -L <llama.cpp>/build/bin -lllama -Wl,-rpath,<llama.cpp>/build/bin
```

Needs ollama with the store's embedding model for the catalog. Env: `MODEL` (gguf path),
`KVT=q8_0`, `NCTX=4096`, `SEL=3`, `GAP=0.04`, `THRESH=0.65` (nomic scale; see reembed section),
`GEN=96`, `DECAY=0.9`, `COPYB=2.0`, `EXTRACT=1` / `GROUND=1` (text answer modes), `MINCH=300`,
`EMB=<ollama model>` (new stores / reembed), `TEXTONLY=1` (catalog+embs only, no KV — log stores),
`EMB_CPU=1` (chat keeps the LLM resident, so the embedder runs on CPU to avoid fighting it for VRAM).

## Chat notes

- The model loads **once** per chat session; each turn (user line + answer) becomes one segment,
  committed immediately (KV spill + catalog + embedding + manifest) — a crash loses nothing.
- On startup the tail segment of the store is restored, so the conversation resumes adjacent to
  existing KV cells (decoding far from any cell in a fresh cache fails — M1 lesson).
- Recall: each user line probes the whole catalog; confident hits that are not already in the
  live window are restored before answering and dropped from the live window after (disk keeps them).
- The live window is capped at `NCTX/2`; older turns are evicted from the cache (already on disk).

## Defrag / migration

`defrag` re-decodes every segment's stored text with the **current** `MODEL` into a fresh store
(`<store>.new`), carries the living weights over, copies the embeddings (they depend only on the
text), then atomically swaps; the old store stays at `<store>.bak`. This is the escape hatch for
all three KV locks: model change, quant change, llama.cpp build change.

## serve (resident daemon)

One-shot `ask` reloads the model each call (~seconds). `kvmem serve <store>` keeps it resident
and answers over plain HTTP on `127.0.0.1` (localhost only, no dependencies):

```bash
KVPORT=8345 kvmem serve mystore &
curl -s -d "what's the gate code?" localhost:8345/ask     # ~0.3–0.5 s; header X-Low-Confidence: 1 if unknown
curl -s localhost:8345/stats
curl -s -X POST localhost:8345/shutdown                    # or SIGTERM — both save weights
```

## prune (cap the store)

`kvmem prune <store> keep=N` (keep the N hottest) or `below=W` (drop weight < W) sinks cold
segments: their KV is deleted and their **text is moved to `archive.tsv`** (never lost — re-ingest
from there if needed). Run `kvmem defrag` afterwards to compact the freed token space — that is
when defrag genuinely shrinks the store (e.g. 25 195 → 344 tokens after pruning to the 10 hottest).

## Status

M6 of the session-memory driver: ingest / ask / chat (ask-fallback) / serve / prune / defrag /
reembed / stats, hybrid probe, `EXTRACT` (verbatim) + `GROUND` (own words, evidence-in-prompt)
modes, per-store embedder (bge-m3 fixes RU). Verified on Qwen2.5-7B-1M: a solid fact store
(generation), reliable log retrieval (`EXTRACT`) and grounded log answers (`GROUND`). Open: a
self-embedding probe channel, OpenAI-style API, 4-bit cold store.
