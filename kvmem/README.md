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

**Boundary (what kvmem does *not* do yet):** recall over messy free-form text (e.g. raw chat
logs) is unreliable. On planted, self-contained facts the probe lands the right segment (cosine
~0.93) and the answer is grounded. On conversational logs the single-segment nomic cosine ranks
*topically* related segments (~0.66) rather than the answer-bearing one, and generation then
confabulates from the model's prior. kvmem today is a **fact store** (notes, configs, structured
knowledge), not a session-recall engine over arbitrary logs — the latter needs a stronger
retriever (hybrid probe) and grounded generation.

## Store layout (`mystore/`)

```
manifest.txt     model_file, model_bytes, kv_type, build, total_tokens, n_segments
catalog.tsv      id  lo  hi  w  text          (source text kept next to KV = re-decode insurance)
embs.f32         float32[n_segments][768]      (nomic embeddings for the probe)
segments/N.kv    serialized KV range for segment N (llama_state_seq; build-sensitive)
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

Needs ollama with `nomic-embed-text` for the catalog. Env: `MODEL` (gguf path), `KVT=q8_0`,
`NCTX=4096`, `SEL=3`, `GAP=0.04`, `THRESH=0.65`, `GEN=96`, `DECAY=0.9`, `COPYB=2.0`,
`EMB_CPU=1` (chat keeps the LLM resident, so nomic embeds on CPU to avoid fighting it for VRAM).

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

M4 of the session-memory driver: ingest / ask / chat (with ask-fallback) / serve / prune /
defrag / stats, plus a Hermes skill. Verified as a fact store on Qwen2.5-7B-1M. Not solved:
recall over free-form logs (see Boundary above — needs a hybrid retriever), OpenAI-style API,
true 4-bit cold store.
