// PoC #18b — living-KV on llama.cpp, step 2: segment ORCHESTRATOR (spill-all + catalog + selective restore).
// Pipeline: chunked prefill -> spill every segment's KV to host store (seq_cp+get_data) ->
//           catalog = nomic embeddings via ollama (external embedder, poc7 lineage) ->
//           budget answer = restore top-SEL segments (set_data) + recency RKEEP -> decode question.
// Runs: A control (full cache) | H hole (recency only) | R random restore | S semantic restore.
// PASS = S rank ~= A rank ~= 0, H and R degraded. Model: Qwen2.5-7B-Instruct-1M Q4_K_M (poc17e's recall champion).
#include "llama.h"
#include <cstdio>
#include <cstdlib>
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

static llama_context *g_ctx;
static float *decode_range(const std::vector<llama_token> &toks, int lo, int hi, llama_pos pos0, llama_seq_id seq, bool want_logits) {
    llama_batch b = llama_batch_init(hi - lo, 0, 1);
    for (int i = lo; i < hi; i++) {
        b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = pos0 + (i - lo);
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = seq;
        b.logits[b.n_tokens] = (want_logits && i == hi - 1) ? 1 : 0;
        b.n_tokens++;
    }
    if (llama_decode(g_ctx, b) != 0) { fprintf(stderr, "decode failed @%d\n", lo); exit(1); }
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

// ---- external embedder: ollama nomic via curl (catalog probe; mid-layer hidden not exposed by llama.cpp)
static std::vector<float> embed(const std::string &text, bool query) {
    std::string esc; for (char c : text) { if (c == '"' || c == '\\') esc += '\\'; esc += c; }
    std::string cmd = "curl -s localhost:11434/api/embeddings -d '{\"model\":\"nomic-embed-text\",\"prompt\":\"";
    cmd += query ? "search_query: " : "search_document: "; cmd += esc + "\"}'";
    FILE *p = popen(cmd.c_str(), "r");
    std::string resp; char buf[4096]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, p)) > 0) resp.append(buf, n);
    pclose(p);
    std::vector<float> v; const char *s = strstr(resp.c_str(), "\"embedding\":[");
    if (!s) { fprintf(stderr, "embed failed: %s\n", resp.substr(0, 200).c_str()); exit(1); }
    s += 13; char *e;
    while (*s && *s != ']') { v.push_back((float)strtod(s, &e)); s = (*e == ',') ? e + 1 : e; }
    return v;
}
static float cosine(const std::vector<float> &a, const std::vector<float> &b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return (float)(d / (sqrt(na)*sqrt(nb) + 1e-9));
}

struct Seg { int lo, hi; std::string text; std::vector<uint8_t> kv; std::vector<float> emb; };

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/data/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";
    const int N      = argc > 2 ? atoi(argv[2]) : 8192;
    const int CHUNK  = 512, RKEEP = 32, SEL = 2;

    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = N + 64; cp.n_batch = CHUNK; cp.n_seq_max = 4;
    cp.kv_unified = true; // seq_cp needs full KV buffer (poc18a lesson)
    g_ctx = llama_init_from_model(model, cp);
    if (!g_ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(g_ctx);

    // ---- corpus (poc17e methodology): segment = sentence, track spans + text
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
        segs.push_back({(int)ids.size(), (int)(ids.size() + tk.size()), t, {}, {}});
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
    printf("N=%d segs=%zu fact=seg%d[%d,%d) ans_tok=%d\n", N, segs.size(), FACTSEG, segs[FACTSEG].lo, segs[FACTSEG].hi, ans_tok);

    // ---- chunked prefill into seq0
    for (int i = 0; i < N; i += CHUNK) decode_range(ids, i, std::min(i + CHUNK, N), i, 0, false);
    printf("prefill done\n");

    // ---- spill: copy every segment's KV range to host store (cache stays intact for control run)
    size_t store_bytes = 0;
    for (auto &s : segs) {
        llama_memory_seq_cp(mem, 0, 1, s.lo, s.hi);
        size_t sz = llama_state_seq_get_size(g_ctx, 1);
        s.kv.resize(sz);
        if (llama_state_seq_get_data(g_ctx, s.kv.data(), sz, 1) != sz) { fprintf(stderr, "get_data failed\n"); return 1; }
        llama_memory_seq_rm(mem, 1, -1, -1);
        store_bytes += sz;
    }
    printf("spilled %zu segs -> %.1f MB host store\n", segs.size(), store_bytes / 1e6);

    // ---- catalog: embed segments + query (external nomic)
    for (auto &s : segs) s.emb = embed(s.text, false);
    auto qemb = embed(QTXT, true);
    std::vector<std::pair<float,int>> scored;
    for (size_t i = 0; i < segs.size(); i++) scored.push_back({cosine(qemb, segs[i].emb), (int)i});
    std::sort(scored.rbegin(), scored.rend());
    printf("catalog top-4: "); for (int i = 0; i < 4; i++) printf("seg%d(%.3f) ", scored[i].second, scored[i].first); printf("\n");

    std::vector<float> logitsA(n_vocab); int rank; double prob;
    auto answer = [&](const char *tag) {
        float *lg = decode_range(qids, 0, (int)qids.size(), N, 0, true);
        rank_prob(lg, n_vocab, ans_tok, &rank, &prob);
        if (!strcmp(tag, "A")) memcpy(logitsA.data(), lg, n_vocab * sizeof(float));
        printf("%s: ans rank %6d  prob %.4f\n", tag, rank, prob);
        llama_memory_seq_rm(mem, 0, N, -1); // drop question tokens for next run
    };
    auto restore_seg = [&](int i) {
        if (llama_state_seq_set_data(g_ctx, segs[i].kv.data(), segs[i].kv.size(), 1) == 0) { fprintf(stderr, "set_data failed\n"); exit(1); }
        llama_memory_seq_cp(mem, 1, 0, -1, -1);
        llama_memory_seq_rm(mem, 1, -1, -1);
    };

    answer("A control (full cache)          ");
    llama_memory_seq_rm(mem, 0, 0, N - RKEEP);   // the big hole: only recency stays in VRAM
    answer("H hole    (recency only)        ");
    int r1 = 3, r2 = 4;                           // fixed non-fact distractor segs
    restore_seg(r1); restore_seg(r2);
    answer("R random  (2 wrong segs+recency)");
    llama_memory_seq_rm(mem, 0, segs[r1].lo, segs[r1].hi);
    llama_memory_seq_rm(mem, 0, segs[r2].lo, segs[r2].hi);
    for (int i = 0; i < SEL; i++) restore_seg(scored[i].second);
    answer("S semantic(top-2 segs+recency)  ");

    bool fact_in = false; for (int i = 0; i < SEL; i++) if (scored[i].second == FACTSEG) fact_in = true;
    printf("fact seg in semantic top-%d: %s\n", SEL, fact_in ? "YES" : "NO");
    llama_free(g_ctx); llama_model_free(model);
    return 0;
}
// compile:
// g++ -O2 -o poc18b poc18b_orchestrator.cpp -I ~/llama.cpp/include -I ~/llama.cpp/ggml/include \
//     -L ~/llama.cpp/build/bin -lllama -Wl,-rpath,$HOME/llama.cpp/build/bin
