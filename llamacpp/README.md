# living-kv on llama.cpp (poc18 series)

Port of the living-KV mechanics (decay / recharge / spill / revival) from the python PoC
(`../living_kv`) onto stock llama.cpp — **no kernel fork**, public C API only
(`llama_memory_seq_cp/rm` + `llama_state_seq_get/set_data`). All "liveness" happens at
segment level in an orchestrator outside attention.

Key prerequisite discovered: `ctx_params.kv_unified = true` (with split KV streams
`seq_cp` aborts: "seq_cp() is only supported for full KV buffers").

## PoCs (all PASS, single day, RTX 5060 8GB, Qwen2.5-7B-Instruct-1M Q4_K_M)

| PoC | What | Result |
|---|---|---|
| 18a | range cut/restore primitive | bit-exact (max logit diff 0.000000) |
| 18b | spill-all + external catalog (nomic) + selective restore | rank 0, p 0.985 @ 0.5% budget; controls: full p 0.881, hole r853, random r275 |
| 18c | streaming prefill through a small buffer | 131k tokens through n_ctx=4096 (32× smaller): rank 0, prefill ~2900 tok/s |
| 18d | w-dynamics: decay / recharge / eviction / revival | repeat query = 0 KB restore (hot hit); evicted fact revived from RAM, rank 0 |
| 18e | quantized cold store (`type_k/type_v` + FA, inside 18c) | q8_0 lossless (@131k p 0.967 = f16, store 7.5→4.0 GB); q4_0 degrades confidence 3× |
| 18f | honest same-type multi-needle (8 needles) | catalog seg-hit 8/8 (python probe was 0/8); SEL=1 → 8/8 rank 0; lesson: restoring a *similar* segment distracts generation → selection must be confidence-aware |
| 18g | decode bench, budget vs full cache | budget **constant ~84 tok/s @ 32k–131k**; full cache 70→62 tok/s and OOM at 65k+ |

## One-command install

```bash
curl -fsSL https://raw.githubusercontent.com/helgard-orlm/living-kv-cache/master/llamacpp/install.sh | bash
```

Clones llama.cpp (pinned to the tested tag), builds it (CUDA if present, else CPU), builds the demos,
downloads a small demo model and runs the poc18a smoke test. `FULL=1 bash install.sh` also pulls the
real Qwen2.5-7B-Instruct-1M model (~4.7 GB). poc18b..g additionally need ollama + `nomic-embed-text`
for the catalog (the script tells you).

## Build (manual)

```
g++ -O2 -o poc18X poc18X_*.cpp -I <llama.cpp>/include -I <llama.cpp>/ggml/include \
    -L <llama.cpp>/build/bin -lllama -Wl,-rpath,<llama.cpp>/build/bin
```

Tested against llama.cpp b9297. Catalog embeddings: `nomic-embed-text` via local ollama
(`/api/embeddings`, `search_document:` / `search_query:` prefixes).

Note: `llama_state_seq_*` serialization format is not guaranteed stable across llama.cpp
builds — persistent stores must record the build tag and keep source text for re-decode.
