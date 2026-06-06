// PoC #18d — living-KV on llama.cpp, step 4: W-DYNAMICS (decay / recharge / eviction / revival).
// Segment weight w: cools every turn (w *= DECAY), recharges when the probe selects it (w += score).
// Budget B hot segments stay resident in VRAM; cold ones are evicted to the host store (copy kept
// from streaming spill); the probe can revive an evicted segment = reactivation (evidence > cooled prior).
// 6-turn scenario over 3 facts: pwd, pwd, wifi, city, city, pwd.
// PASS = all turns rank 0; turn2 restore_bytes == 0 (hot hit); fact1 evicted mid-run; turn6 revival works.
// Honest note: recharge signal = probe selection score (segment level); python-poc used real attention —
// llama.cpp does not expose attention weights, this is the available proxy.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

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

struct Seg { int lo, hi; std::string text; std::vector<uint8_t> kv; std::vector<float> emb;
             bool spilled = false, resident = false; float w = 0; };

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/data/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";
    const int N     = argc > 2 ? atoi(argv[2]) : 8192;
    const int NCTX  = 2048, CHUNK = 256, WIN = 256, SELM = 2, BUDGET = 3;
    const float DECAY = 0.7f;

    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = NCTX; cp.n_batch = CHUNK; cp.n_seq_max = 4;
    cp.kv_unified = true;
    g_ctx = llama_init_from_model(model, cp);
    if (!g_ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(g_ctx);

    // ---- corpus: 3 facts spread through distractors
    struct Fact { std::string text, q, ans; int seg = -1; };
    Fact F[3] = {
        {" Important: my computer password is bluefalcon.", " Now, what is my computer password? My computer password is", " blue"},
        {" Remember: the wifi access code is sunsetpark.",  " Now, what is the wifi access code? The wifi access code is", " sunset"},
        {" Note: my cousin Maria lives in Lisbon.",         " Now, in which city does my cousin Maria live? Maria lives in", " Lisbon"}};
    const char *DIST[6] = {
        " The weather was grey and damp with drizzle and a cold breeze at night.",
        " Boats rocked against the wooden pier under a pale quiet sky and gulls called.",
        " People walked past the cafe near the corner talking quietly about nothing.",
        " On Monday it rained, by Tuesday the roads were dry, Wednesday turned warm.",
        " The shelves were full of dusty books that nobody had opened in many years.",
        " My old password used to be redhawk before I changed it a long time ago."};

    std::vector<llama_token> ids; std::vector<Seg> segs;
    auto add = [&](const std::string &t, bool first) -> int {
        auto tk = tokenize(vocab, t, first);
        segs.push_back({(int)ids.size(), (int)(ids.size() + tk.size()), t, {}, {}});
        ids.insert(ids.end(), tk.begin(), tk.end());
        return (int)segs.size() - 1;
    };
    add(" Notes.", true);
    F[0].seg = add(F[0].text, false);                    // pwd near start
    int di = 0;
    while ((int)ids.size() < N / 2) add(DIST[di++ % 6], false);
    F[1].seg = add(F[1].text, false);                    // wifi mid
    while ((int)ids.size() < 3 * N / 4) add(DIST[di++ % 6], false);
    F[2].seg = add(F[2].text, false);                    // city at 3/4
    while ((int)ids.size() < N) add(DIST[di++ % 6], false);
    while ((int)ids.size() > N) ids.pop_back();
    while (!segs.empty() && segs.back().lo >= N) segs.pop_back();
    segs.back().hi = std::min(segs.back().hi, N);
    printf("N=%d n_ctx=%d segs=%zu facts: pwd=seg%d wifi=seg%d city=seg%d | B=%d decay=%.1f\n",
           N, NCTX, segs.size(), F[0].seg, F[1].seg, F[2].seg, BUDGET, DECAY);

    // ---- streaming prefill with spill (poc18c)
    size_t next_unspilled = 0; int kept_from = 0;
    for (int i = 0; i < N; i += CHUNK) {
        int e = std::min(i + CHUNK, N);
        decode_range(ids, i, e, i, 0, false);
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
    printf("streamed, %zu segs spilled, tail [%d,%d) resident as recency\n\n", next_unspilled, kept_from, N);

    // ---- 6-turn living loop: pwd, pwd, wifi, city, city, pwd
    int plan[6] = {0, 0, 1, 2, 2, 0};
    printf("turn  ask   rank   prob    restore_KB  resident_after(w)        w[pwd] w[wifi] w[city]\n");
    bool evicted_pwd_seen = false; size_t t2_restore = 1; int fails = 0;
    for (int t = 0; t < 6; t++) {
        Fact &f = F[plan[t]];
        auto qids = tokenize(vocab, f.q, false);
        llama_token ans_tok = tokenize(vocab, f.ans, false)[0];
        // probe over spilled segments (resident ones score too — selection may be a free hot hit)
        auto qemb = embed(f.q, true);
        std::vector<std::pair<float,int>> scored;
        for (size_t i = 0; i < segs.size(); i++) if (segs[i].spilled) {
            if (segs[i].emb.empty()) segs[i].emb = embed(segs[i].text, false);
            scored.push_back({cosine(qemb, segs[i].emb), (int)i});
        }
        std::sort(scored.rbegin(), scored.rend());
        // decay all, recharge selected, restore the non-resident selected
        for (auto &s : segs) s.w *= DECAY;
        size_t restore_bytes = 0;
        for (int k = 0; k < SELM; k++) {
            Seg &s = segs[scored[k].second]; s.w += scored[k].first;
            if (!s.resident) {
                if (llama_state_seq_set_data(g_ctx, s.kv.data(), s.kv.size(), 1) == 0) { fprintf(stderr, "set_data failed\n"); return 1; }
                llama_memory_seq_cp(mem, 1, 0, -1, -1);
                llama_memory_seq_rm(mem, 1, -1, -1);
                s.resident = true; restore_bytes += s.kv.size();
            }
        }
        if (t == 1) t2_restore = restore_bytes;
        // answer
        float *lg = decode_range(qids, 0, (int)qids.size(), N, 0, true);
        int rank; double prob; rank_prob(lg, n_vocab, ans_tok, &rank, &prob);
        if (rank != 0) fails++;
        llama_memory_seq_rm(mem, 0, N, -1);
        // budget: keep top-BUDGET resident by w, evict the rest (host copy already in store)
        std::vector<int> res;
        for (size_t i = 0; i < segs.size(); i++) if (segs[i].resident) res.push_back((int)i);
        std::sort(res.begin(), res.end(), [&](int a, int b){ return segs[a].w > segs[b].w; });
        for (size_t k = BUDGET; k < res.size(); k++) {
            Seg &s = segs[res[k]];
            llama_memory_seq_rm(mem, 0, s.lo, s.hi);
            s.resident = false;
            if (res[k] == F[0].seg) evicted_pwd_seen = true;
        }
        res.resize(std::min(res.size(), (size_t)BUDGET));
        std::string rs; for (int i : res) { char b[48]; snprintf(b, 48, "seg%d(%.2f) ", i, segs[i].w); rs += b; }
        printf("T%d    %-5s %5d  %.4f  %9.0f  %-24s %.2f   %.2f    %.2f\n",
               t + 1, plan[t] == 0 ? "pwd" : plan[t] == 1 ? "wifi" : "city",
               rank, prob, restore_bytes / 1024.0, rs.c_str(), segs[F[0].seg].w, segs[F[1].seg].w, segs[F[2].seg].w);
    }
    printf("\nchecks: all_ranks_0=%s  t2_free_hot_hit=%s  pwd_evicted_midrun=%s\n",
           fails == 0 ? "YES" : "NO", t2_restore == 0 ? "YES" : "NO", evicted_pwd_seen ? "YES" : "NO");
    printf(fails == 0 && t2_restore == 0 && evicted_pwd_seen ? "PASS: living dynamics work (hot hit + eviction + revival)\n"
                                                             : "CHECK: some dynamic failed\n");
    llama_free(g_ctx); llama_model_free(model);
    return 0;
}
// compile:
// g++ -O2 -o poc18d poc18d_living.cpp -I ~/llama.cpp/include -I ~/llama.cpp/ggml/include \
//     -L ~/llama.cpp/build/bin -lllama -Wl,-rpath,$HOME/llama.cpp/build/bin
