// PoC #18f — living-KV on llama.cpp, step 6: HONEST same-type MULTI-NEEDLE (port of python poc17 B).
// Python finding: 8 same-template needles defeat budget selection (semseg topical probe 0/8 seg-hits,
// rawqk mean rank 889; flat cache = 0 ideal). The catalog here is DIFFERENT: external nomic text
// embeddings, where entity names are a strong lexical signal. Question: does the weak spot port,
// or does the external-embedder catalog close it?
// 8 needles "Fact: <Name> works in <City>." spread through distractors; query each by name.
// Report per-needle: probe seg-hit + answer rank. No pre-declared PASS — honest replication.
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

struct Seg { int lo, hi; std::string text; std::vector<uint8_t> kv; std::vector<float> emb; bool spilled = false; };

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/data/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";
    const int N    = argc > 2 ? atoi(argv[2]) : 8192;
    const int NCTX = 2048, CHUNK = 256, WIN = 256, NF = 8;
    const int SEL = argc > 3 ? atoi(argv[3]) : 2;

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

    const char *NAME[NF] = {"Marcus", "Elena", "Viktor", "Sofia", "Henrik", "Amara", "Diego", "Yuki"};
    const char *CITY[NF] = {"Toronto", "Madrid", "Cairo", "Oslo", "Lima", "Prague", "Dublin", "Athens"};
    const char *DIST[6] = {
        " The weather was grey and damp with drizzle and a cold breeze at night.",
        " Boats rocked against the wooden pier under a pale quiet sky and gulls called.",
        " People walked past the cafe near the corner talking quietly about nothing.",
        " On Monday it rained, by Tuesday the roads were dry, Wednesday turned warm.",
        " The shelves were full of dusty books that nobody had opened in many years.",
        " Last year a colleague mentioned moving abroad for an office job somewhere."};

    std::vector<llama_token> ids; std::vector<Seg> segs; int fseg[NF];
    auto add = [&](const std::string &t, bool first) -> int {
        auto tk = tokenize(vocab, t, first);
        segs.push_back({(int)ids.size(), (int)(ids.size() + tk.size()), t, {}, {}});
        ids.insert(ids.end(), tk.begin(), tk.end());
        return (int)segs.size() - 1;
    };
    add(" Notes.", true);
    int di = 0;
    for (int f = 0; f < NF; f++) {            // needle f at ~(f+0.5)/NF of the stream
        int target = (int)((long long)N * (2 * f + 1) / (2 * NF));
        while ((int)ids.size() < target) add(DIST[di++ % 6], false);
        char nb[96]; snprintf(nb, 96, " Fact: %s works in %s.", NAME[f], CITY[f]);
        fseg[f] = add(nb, false);
    }
    while ((int)ids.size() < N) add(DIST[di++ % 6], false);
    while ((int)ids.size() > N) ids.pop_back();
    while (!segs.empty() && segs.back().lo >= N) segs.pop_back();
    segs.back().hi = std::min(segs.back().hi, N);
    printf("N=%d segs=%zu needles at segs:", N, segs.size());
    for (int f = 0; f < NF; f++) printf(" %d", fseg[f]); printf("\n");

    // ---- streaming prefill with spill (poc18c skeleton)
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
    printf("streamed, %zu spilled\n\n", next_unspilled);

    // ---- query each needle: probe -> restore top-SEL -> answer -> evict restored (clean next query)
    printf("query        probe_top2            seg_hit  ans_rank  prob\n");
    int seghits = 0, rank0 = 0; long ranksum = 0;
    for (int f = 0; f < NF; f++) {
        char qb[160], ab[64];
        snprintf(qb, 160, " Now, in which city does %s work? %s works in", NAME[f], NAME[f]);
        snprintf(ab, 64, " %s", CITY[f]);
        auto qids = tokenize(vocab, qb, false);
        llama_token ans_tok = tokenize(vocab, ab, false)[0];
        auto qemb = embed(qb, true);
        std::vector<std::pair<float,int>> scored;
        for (size_t i = 0; i < segs.size(); i++) if (segs[i].spilled) {
            if (segs[i].emb.empty()) segs[i].emb = embed(segs[i].text, false);
            scored.push_back({cosine(qemb, segs[i].emb), (int)i});
        }
        std::sort(scored.rbegin(), scored.rend());
        bool hit = false;
        for (int k = 0; k < SEL; k++) {
            int si = scored[k].second; if (si == fseg[f]) hit = true;
            if (llama_state_seq_set_data(g_ctx, segs[si].kv.data(), segs[si].kv.size(), 1) == 0) { fprintf(stderr, "set_data failed\n"); return 1; }
            llama_memory_seq_cp(mem, 1, 0, -1, -1);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
        if (hit) seghits++;
        float *lg = decode_range(qids, 0, (int)qids.size(), N, 0, true);
        int rank; double prob; rank_prob(lg, n_vocab, ans_tok, &rank, &prob);
        if (rank == 0) rank0++; ranksum += rank;
        llama_memory_seq_rm(mem, 0, N, -1);
        for (int k = 0; k < SEL; k++) { Seg &s = segs[scored[k].second]; llama_memory_seq_rm(mem, 0, s.lo, s.hi); }
        printf("%-12s seg%d(%.3f) seg%d(%.3f)  %-7s %8d  %.4f\n", NAME[f],
               scored[0].second, scored[0].first, scored[1].second, scored[1].first,
               hit ? "YES" : "NO", rank, prob);
    }
    printf("\nsame-type multi-needle: seg-hit %d/%d | rank0 %d/%d | mean rank %.1f\n",
           seghits, NF, rank0, NF, (double)ranksum / NF);
    printf("(python poc17B baseline: semseg 0/8 seg-hits, rawqk mean rank 889)\n");
    llama_free(g_ctx); llama_model_free(model);
    return 0;
}
// compile:
// g++ -O2 -o poc18f poc18f_multineedle.cpp -I ~/llama.cpp/include -I ~/llama.cpp/ggml/include \
//     -L ~/llama.cpp/build/bin -lllama -Wl,-rpath,$HOME/llama.cpp/build/bin
