# kvmem — living-KV session memory (ingest once, ask forever)

Feed text into a model **once**; ask questions in a **later, separate process**. The model
recalls from its own stored KV-cache — no re-reading the source, no RAG re-prefill. Built on the
[llama.cpp port](../llamacpp) of living-KV (streaming spill + nomic catalog + w-dynamics + q8_0).

```bash
kvmem ingest mystore notes.txt          # stream text through a small KV buffer, spill to disk
kvmem ingest mystore more.txt           # append (positions continue across files)
kvmem ask    mystore "what's the gate code?"   # fresh process: probe -> restore relevant KV -> answer
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

**Known soft spot:** exact verbatim recall of a *novel multi-token string* (e.g. an invented
hostname `cobalt-finch`) can be paraphrased even though the correct segment is retrieved — the recall
mechanism is solid, the generation's copy-fidelity of rare tokens is not guaranteed. Matches the
"confidence decays" caveat of the parent PoC.

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
`NCTX=4096`, `SEL=3`, `GAP=0.04`, `THRESH=0.65`, `GEN=96`, `DECAY=0.9`.

## Status

M1 (ingest-first) of the session-memory driver. Not yet: interactive chat daemon, defrag/migration
when the model changes, true 4-bit cold store. PoC quality — see the soft spot above.
