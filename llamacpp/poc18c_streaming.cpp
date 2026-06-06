// PoC #18c — living-KV on llama.cpp, step 3: STREAMING prefill with small n_ctx (true VRAM bound).
// n_ctx = 1024 cells, stream N = 8192 tokens through it: rolling window WIN, spill finished segments
// to host store as they leave the window, rm them from the cache. Answer: restore top-SEL segments
// (positions up to N >> n_ctx!) + window -> decode question at pos N.
// OPEN QUESTION this poc answers: do cells with pos >> n_ctx survive in a small unified buffer?
// PASS = recall rank ~0 (matches poc18b S run) with KV buffer 8x smaller than the context length.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text, bool add_special) {
    int n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, add_special, false);
    std::vector<llama_token> out(n);
    llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), out.data(), n, add_special, false);
    return out;
}

static llama_context *g_ctx;
static float *decode_range(const std::vector<llama_token> &toks, int lo, int hi, llama_pos pos0, llama_seq_id seq, bool want_logits) {
    llama_batch b = llama_batch_init(hi - lo, 0, 1);
    for (int i = lo; i < hi; i++) {
        b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = pos0 + (i - lo);
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = seq;
        b.logits[b.n_tokens] = (want_logits && i == hi - 1) ? 1 : 0;
        b.n_tokens++;
    }
    int rc = llama_decode(g_ctx, b);
    if (rc != 0) { fprintf(stderr, "decode failed rc=%d @%d (pos %d)\n", rc, lo, (int)pos0); exit(1); }
    float *lg = want_logits ? llama_get_logits_ith(g_ctx, b.n_tokens - 1) : nullptr;
    llama_batch_free(b);
    return lg;
}

static void rank_prob(const float *logits, int n_vocab, llama_token tid, int *rank, double *prob) {
    float v = logits[tid]; int r = 0; double mx = logits[0];
    for (int i = 0; i < n_vocab; i++) { if (logits[i] > v) r++; if (logits[i] > mx) mx = logits[i]; }
    double z = 0; for (int i = 0; i < n_vocab; i++) z += exp((double)logits[i] - mx);
    *rank = r; *prob = exp((double)v - mx) / z;
}

#include <map>
static std::vector<float> embed_uncached(const std::string &text, bool query);
static std::vector<float> embed(const std::string &text, bool query) {
    static std::map<std::string, std::vector<float>> cache; // distractors repeat -> 1 curl per unique text
    auto key = (query ? "q:" : "d:") + text;
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, embed_uncached(text, query)).first;
    return it->second;
}
static std::vector<float> embed_uncached(const std::string &text, bool query) {
    std::string esc; for (char c : text) { if (c == '"' || c == '\\') esc += '\\'; esc += c; }
    std::string cmd = "curl -s localhost:11434/api/embeddings -d '{\"model\":\"nomic-embed-text\",\"prompt\":\"";
    cmd += query ? "search_query: " : "search_document: "; cmd += esc + "\"}'";
    FILE *p = popen(cmd.c_str(), "r");
    std::string resp; char buf[4096]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, p)) > 0) resp.append(buf, n);
    pclose(p);
    std::vector<float> v; const char *s = strstr(resp.c_str(), "\"embedding\":[");
    if (!s) { fprintf(stderr, "embed failed\n"); exit(1); }
    s += 13; char *e;
    while (*s && *s != ']') { v.push_back((float)strtod(s, &e)); s = (*e == ',') ? e + 1 : e; }
    return v;
}
static float cosine(const std::vector<float> &a, const std::vector<float> &b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return (float)(d / (sqrt(na)*sqrt(nb) + 1e-9));
}

struct Seg { int lo, hi; std::string text; std::vector<uint8_t> kv; std::vector<float> emb; bool spilled = false; };

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/data/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";
    const int N     = argc > 2 ? atoi(argv[2]) : 8192;
    const int NCTX  = argc > 3 ? atoi(argv[3]) : 1024;   // the small buffer — the whole point
    const char *kvt = argc > 4 ? argv[4] : "f16";        // f16 | q8_0 | q4_0 — quantized cold store (poc18e)
    const int CHUNK = 256, WIN = 256, SEL = 2;

    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = NCTX; cp.n_batch = CHUNK; cp.n_seq_max = 4;
    cp.kv_unified = true;
    if (strcmp(kvt, "f16")) { // quantized KV: resident cache AND serialized store shrink together
        cp.type_k = cp.type_v = !strcmp(kvt, "q8_0") ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // quantized V requires FA
    }
    printf("kv type: %s\n", kvt);
    g_ctx = llama_init_from_model(model, cp);
    if (!g_ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(g_ctx);

    const std::string FACT = " Important: my computer password is bluefalcon.";
    const std::string QTXT = " Now, what is my computer password? My computer password is";
    const std::string ANS  = " blue";
    const char *DIST[6] = {
        " The weather was grey and damp with drizzle and a cold breeze at night.",
        " Boats rocked against the wooden pier under a pale quiet sky and gulls called.",
        " People walked past the cafe near the corner talking quietly about nothing.",
        " On Monday it rained, by Tuesday the roads were dry, Wednesday turned warm.",
        " The shelves were full of dusty books that nobody had opened in many years.",
        " My old password used to be redhawk before I changed it a long time ago."};

    std::vector<llama_token> ids; std::vector<Seg> segs;
    auto add = [&](const std::string &t, bool first) {
        auto tk = tokenize(vocab, t, first);
        segs.push_back({(int)ids.size(), (int)(ids.size() + tk.size()), t, {}, {}, false});
        ids.insert(ids.end(), tk.begin(), tk.end());
    };
    add(" Notes.", true); add(FACT, false);
    for (int di = 0; (int)ids.size() < N; di++) add(DIST[di % 6], false);
    while ((int)ids.size() > N) ids.pop_back();
    while (!segs.empty() && segs.back().lo >= N) segs.pop_back();
    segs.back().hi = std::min(segs.back().hi, N);
    const int FACTSEG = 1;
    auto qids = tokenize(vocab, QTXT, false);
    llama_token ans_tok = tokenize(vocab, ANS, false)[0];
    printf("N=%d n_ctx=%d (%.0fx smaller) segs=%zu fact=seg%d\n", N, NCTX, (double)N/NCTX, segs.size(), FACTSEG);

    // ---- streaming prefill: decode chunk -> spill segments that left the window -> rm them
    auto t0 = std::chrono::steady_clock::now();
    size_t store_bytes = 0; size_t next_unspilled = 0; int kept_from = 0;
    for (int i = 0; i < N; i += CHUNK) {
        int e = std::min(i + CHUNK, N);
        decode_range(ids, i, e, i, 0, false);
        int cutoff = e - WIN;
        while (next_unspilled < segs.size() && segs[next_unspilled].hi <= cutoff) {
            Seg &s = segs[next_unspilled];
            llama_memory_seq_cp(mem, 0, 1, s.lo, s.hi);
            size_t sz = llama_state_seq_get_size(g_ctx, 1);
            s.kv.resize(sz);
            if (llama_state_seq_get_data(g_ctx, s.kv.data(), sz, 1) != sz) { fprintf(stderr, "get_data failed seg%zu\n", next_unspilled); return 1; }
            llama_memory_seq_rm(mem, 1, -1, -1);
            s.spilled = true; store_bytes += sz; next_unspilled++;
        }
        int rm_to = next_unspilled < segs.size() ? std::min(cutoff, segs[next_unspilled].lo) : cutoff;
        if (rm_to > kept_from) { llama_memory_seq_rm(mem, 0, kept_from, rm_to); kept_from = rm_to; }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("streamed: %zu/%zu segs spilled -> %.1f MB host; cache holds [%d,%d); prefill %.1fs (%.0f tok/s)\n",
           next_unspilled, segs.size(), store_bytes / 1e6, kept_from, N, secs, N / secs);

    // ---- catalog + selective restore (only spilled segs are candidates; tail segs are still resident)
    auto qemb = embed(QTXT, true);
    std::vector<std::pair<float,int>> scored;
    for (size_t i = 0; i < segs.size(); i++) if (segs[i].spilled) {
        segs[i].emb = embed(segs[i].text, false);
        scored.push_back({cosine(qemb, segs[i].emb), (int)i});
    }
    std::sort(scored.rbegin(), scored.rend());
    printf("catalog top-3: "); for (int i = 0; i < 3; i++) printf("seg%d(%.3f) ", scored[i].second, scored[i].first); printf("\n");
    bool fact_in = false;
    for (int i = 0; i < SEL; i++) {
        int si = scored[i].second; if (si == FACTSEG) fact_in = true;
        if (llama_state_seq_set_data(g_ctx, segs[si].kv.data(), segs[si].kv.size(), 1) == 0) { fprintf(stderr, "set_data failed\n"); return 1; }
        llama_memory_seq_cp(mem, 1, 0, -1, -1);
        llama_memory_seq_rm(mem, 1, -1, -1);
    }

    // ---- answer at pos N (cells now span pos 5..N with huge holes, buffer is only NCTX cells)
    float *lg = decode_range(qids, 0, (int)qids.size(), N, 0, true);
    int rank; double prob; rank_prob(lg, n_vocab, ans_tok, &rank, &prob);
    printf("S streaming: ans rank %d  prob %.4f  fact in top-%d: %s\n", rank, prob, SEL, fact_in ? "YES" : "NO");
    printf(rank == 0 ? "PASS: VRAM-bounded streaming recall works (n_ctx %dx smaller than N)\n" : "CHECK: recall degraded\n", N/NCTX);
    llama_free(g_ctx); llama_model_free(model);
    return 0;
}
// compile:
// g++ -O2 -o poc18c poc18c_streaming.cpp -I ~/llama.cpp/include -I ~/llama.cpp/ggml/include \
//     -L ~/llama.cpp/build/bin -lllama -Wl,-rpath,$HOME/llama.cpp/build/bin
