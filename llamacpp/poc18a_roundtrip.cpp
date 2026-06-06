// PoC #18a — living-KV on llama.cpp, step 1: range spill ROUNDTRIP primitive.
// Question: does seq_cp(range) -> state_seq_get_data -> seq_rm(range) -> state_seq_set_data
// restore KV cells at their ORIGINAL positions so the model doesn't notice?
// Test: A control (full cache) vs B hole (fact removed) vs C restored (fact cut to host buffer, put back).
// PASS = C answer rank == A answer rank (0) AND logits diff tiny; B rank must degrade (proves rm is real).
// Model: tinyllama Q4 (mechanics only). Build: see compile line at bottom.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text, bool add_special) {
    int n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, add_special, false);
    std::vector<llama_token> out(n);
    llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), out.data(), n, add_special, false);
    return out;
}

// decode a token range [pos0..) into seq, request logits on last token if want_logits
static float *decode_tokens(llama_context *ctx, const std::vector<llama_token> &toks, llama_pos pos0, llama_seq_id seq, bool want_logits) {
    llama_batch b = llama_batch_init((int32_t)toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[b.n_tokens]    = toks[i];
        b.pos[b.n_tokens]      = pos0 + (llama_pos)i;
        b.n_seq_id[b.n_tokens] = 1;
        b.seq_id[b.n_tokens][0] = seq;
        b.logits[b.n_tokens]   = (want_logits && i == toks.size() - 1) ? 1 : 0;
        b.n_tokens++;
    }
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
    float *logits = want_logits ? llama_get_logits_ith(ctx, b.n_tokens - 1) : nullptr;
    llama_batch_free(b);
    return logits;
}

static int rank_of(const float *logits, int n_vocab, llama_token tid) {
    float v = logits[tid]; int r = 0;
    for (int i = 0; i < n_vocab; i++) if (logits[i] > v) r++;
    return r;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/data/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_seq_max = 4;
    cp.kv_unified = true; // seq_cp() requires a full (unified) KV buffer — split streams abort
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    // --- build prompt: notes + FACT + distractors, track fact token span (poc17e methodology)
    const std::string FACT = " Important: my computer password is bluefalcon.";
    const std::string Q    = " Now, what is my computer password? My computer password is";
    const std::string ANS  = " blue";
    const char *DIST[6] = {
        " The weather was grey and damp with drizzle and a cold breeze at night.",
        " Boats rocked against the wooden pier under a pale quiet sky and gulls called.",
        " People walked past the cafe near the corner talking quietly about nothing.",
        " On Monday it rained, by Tuesday the roads were dry, Wednesday turned warm.",
        " The shelves were full of dusty books that nobody had opened in many years.",
        " My old password used to be redhawk before I changed it a long time ago."};

    std::vector<llama_token> ids; int fact_lo = 0, fact_hi = 0;
    { auto t = tokenize(vocab, " Notes.", true);  ids.insert(ids.end(), t.begin(), t.end()); }
    { auto t = tokenize(vocab, FACT, false); fact_lo = (int)ids.size(); ids.insert(ids.end(), t.begin(), t.end()); fact_hi = (int)ids.size(); }
    for (int di = 0; (int)ids.size() < 400; di++) { auto t = tokenize(vocab, DIST[di % 6], false); ids.insert(ids.end(), t.begin(), t.end()); }
    const int N = (int)ids.size();
    auto qids = tokenize(vocab, Q, false);
    llama_token ans_tok = tokenize(vocab, ANS, false)[0];
    printf("N=%d fact_span=[%d,%d) q=%zu ans_tok=%d\n", N, fact_lo, fact_hi, qids.size(), ans_tok);

    std::vector<float> logitsA(n_vocab);

    // ---- run A: control
    llama_memory_clear(mem, true);
    decode_tokens(ctx, ids, 0, 0, false);
    { float *lg = decode_tokens(ctx, qids, N, 0, true);
      memcpy(logitsA.data(), lg, n_vocab * sizeof(float));
      printf("A control : ans rank %d\n", rank_of(lg, n_vocab, ans_tok)); }

    // ---- run B: hole (fact removed, NOT restored)
    llama_memory_clear(mem, true);
    decode_tokens(ctx, ids, 0, 0, false);
    if (!llama_memory_seq_rm(mem, 0, fact_lo, fact_hi)) { fprintf(stderr, "seq_rm failed\n"); return 1; }
    { float *lg = decode_tokens(ctx, qids, N, 0, true);
      printf("B hole    : ans rank %d  (must be >> 0)\n", rank_of(lg, n_vocab, ans_tok)); }

    // ---- run C: cut fact range to host buffer via seq_cp + state_seq_get_data, then restore
    llama_memory_clear(mem, true);
    decode_tokens(ctx, ids, 0, 0, false);
    llama_memory_seq_cp(mem, 0, 1, fact_lo, fact_hi);              // mark range as seq1 too
    size_t sz = llama_state_seq_get_size(ctx, 1);
    std::vector<uint8_t> buf(sz);
    size_t got = llama_state_seq_get_data(ctx, buf.data(), sz, 1); // serialize seq1 (the range) to host
    printf("C spill   : range [%d,%d) -> %zu bytes host (got %zu)\n", fact_lo, fact_hi, sz, got);
    llama_memory_seq_rm(mem, 1, -1, -1);                           // drop temp seq
    llama_memory_seq_rm(mem, 0, fact_lo, fact_hi);                 // make the hole in seq0
    // restore: deserialize into seq1 (cells land at stored positions), then re-mark as seq0
    size_t setn = llama_state_seq_set_data(ctx, buf.data(), got, 1);
    if (setn == 0) { fprintf(stderr, "set_data failed\n"); return 1; }
    llama_memory_seq_cp(mem, 1, 0, -1, -1);
    llama_memory_seq_rm(mem, 1, -1, -1);
    { float *lg = decode_tokens(ctx, qids, N, 0, true);
      int r = rank_of(lg, n_vocab, ans_tok);
      double md = 0; for (int i = 0; i < n_vocab; i++) md = std::max(md, (double)std::fabs(lg[i] - logitsA[i]));
      printf("C restored: ans rank %d  max|logit diff vs A| %.6f\n", r, md);
      printf(r == rank_of(logitsA.data(), n_vocab, ans_tok) && md < 0.5 ? "PASS: roundtrip preserves cache\n"
                                                                        : "CHECK: rank or logits drifted\n"); }

    llama_free(ctx); llama_model_free(model);
    return 0;
}
// compile:
// g++ -O2 -o poc18a poc18a_roundtrip.cpp -I ~/llama.cpp/include -I ~/llama.cpp/ggml/include \
//     -L ~/llama.cpp/build/bin -lllama -Wl,-rpath,$HOME/llama.cpp/build/bin
