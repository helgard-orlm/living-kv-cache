// PoC #18g — living-KV on llama.cpp, step 7: DECODE BENCH (generation speed, budget cache vs full cache).
// Python arc reference: living ~126 tok/s CONSTANT 8k->100k vs full-cache 40 tok/s (3.2x), full OOM @131k.
// Modes: full   — plain full-cache prefill (n_ctx = N), generate GEN tokens, measure tok/s
//        budget — poc18c streaming spill (n_ctx = 4096), restore top-SEL + recency, generate, measure
// Expectation: budget tok/s ~constant vs N (attention over <=n_ctx cells); full degrades with N and OOMs early.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <chrono>

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text, bool add_special) {
    int n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, add_special, false);
    std::vector<llama_token> out(n);
    llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), out.data(), n, add_special, false);
    return out;
}

static llama_context *g_ctx;
static float *decode_tok(const llama_token *toks, int n, llama_pos pos0, bool want_logits) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = pos0 + i;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0;
        b.logits[b.n_tokens] = (want_logits && i == n - 1) ? 1 : 0;
        b.n_tokens++;
    }
    if (llama_decode(g_ctx, b) != 0) { fprintf(stderr, "decode failed @pos %d\n", (int)pos0); exit(1); }
    float *lg = want_logits ? llama_get_logits_ith(g_ctx, b.n_tokens - 1) : nullptr;
    llama_batch_free(b);
    return lg;
}

static std::vector<float> embed(const std::string &text, bool query) {
    static std::map<std::string, std::vector<float>> cache;
    auto key = (query ? "q:" : "d:") + text;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
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
    cache[key] = v;
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
    const int N = argc > 2 ? atoi(argv[2]) : 8192;
    const bool budget = argc > 3 && !strcmp(argv[3], "budget");
    const int CHUNK = 512, WIN = 256, SEL = 2, GEN = 64, NCTX_BUDGET = 4096;

    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = budget ? NCTX_BUDGET : N + 128;
    cp.n_batch = CHUNK; cp.n_seq_max = 4; cp.kv_unified = true;
    g_ctx = llama_init_from_model(model, cp);
    if (!g_ctx) { fprintf(stderr, "ctx init failed (OOM at n_ctx=%d?)\n", (int)cp.n_ctx); return 1; }
    llama_memory_t mem = llama_get_memory(g_ctx);

    const std::string FACT = " Important: my computer password is bluefalcon.";
    const std::string QTXT = " Now, what is my computer password? My computer password is";
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

    auto t0 = std::chrono::steady_clock::now();
    if (!budget) {
        for (int i = 0; i < N; i += CHUNK) decode_tok(ids.data() + i, std::min(CHUNK, N - i), i, false);
    } else {
        size_t next_unspilled = 0; int kept_from = 0;
        for (int i = 0; i < N; i += CHUNK) {
            int e = std::min(i + CHUNK, N);
            decode_tok(ids.data() + i, e - i, i, false);
            int cutoff = e - WIN;
            while (next_unspilled < segs.size() && segs[next_unspilled].hi <= cutoff) {
                Seg &s = segs[next_unspilled];
                llama_memory_seq_cp(mem, 0, 1, s.lo, s.hi);
                size_t sz = llama_state_seq_get_size(g_ctx, 1);
                s.kv.resize(sz);
                if (llama_state_seq_get_data(g_ctx, s.kv.data(), sz, 1) != sz) { fprintf(stderr, "get_data failed\n"); return 1; }
                llama_memory_seq_rm(mem, 1, -1, -1);
                s.spilled = true; next_unspilled++;
            }
            int rm_to = next_unspilled < segs.size() ? std::min(cutoff, segs[next_unspilled].lo) : cutoff;
            if (rm_to > kept_from) { llama_memory_seq_rm(mem, 0, kept_from, rm_to); kept_from = rm_to; }
        }
        // restore top-SEL by catalog
        auto qemb = embed(QTXT, true);
        std::vector<std::pair<float,int>> scored;
        for (size_t i = 0; i < segs.size(); i++) if (segs[i].spilled) {
            if (segs[i].emb.empty()) segs[i].emb = embed(segs[i].text, false);
            scored.push_back({cosine(qemb, segs[i].emb), (int)i});
        }
        std::sort(scored.rbegin(), scored.rend());
        for (int k = 0; k < SEL; k++) {
            Seg &s = segs[scored[k].second];
            if (llama_state_seq_set_data(g_ctx, s.kv.data(), s.kv.size(), 1) == 0) { fprintf(stderr, "set_data failed\n"); return 1; }
            llama_memory_seq_cp(mem, 1, 0, -1, -1);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }
    double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // question, then GEN greedy tokens timed
    auto qids = tokenize(vocab, QTXT, false);
    float *lg = decode_tok(qids.data(), (int)qids.size(), N, true);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::string gen_text;
    auto t1 = std::chrono::steady_clock::now();
    llama_pos pos = N + (llama_pos)qids.size();
    for (int g = 0; g < GEN; g++) {
        llama_token best = 0; float bv = lg[0];
        for (int i = 1; i < n_vocab; i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
        char piece[64]; int pn = llama_token_to_piece(vocab, best, piece, sizeof piece, 0, false);
        if (pn > 0 && g < 8) gen_text.append(piece, pn);
        lg = decode_tok(&best, 1, pos++, true);
    }
    double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    printf("RESULT mode=%s N=%d n_ctx=%d | prefill %.1fs (%.0f t/s) | decode %.1f tok/s | first-gen:%s\n",
           budget ? "budget" : "full", N, (int)cp.n_ctx, prefill_s, N / prefill_s, GEN / gen_s, gen_text.c_str());
    llama_free(g_ctx); llama_model_free(model);
    return 0;
}
// compile:
// g++ -O2 -o poc18g poc18g_decodebench.cpp -I ~/llama.cpp/include -I ~/llama.cpp/ggml/include \
//     -L ~/llama.cpp/build/bin -lllama -Wl,-rpath,$HOME/llama.cpp/build/bin
