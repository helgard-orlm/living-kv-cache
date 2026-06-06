// kvmem — living-KV session memory driver (M1, ingest-first).
//
// Feed real text into a model ONCE; ask questions in a LATER process — the model recalls
// from its own stored KV (no re-reading, no RAG re-prefill). Built from the poc18 pieces
// (see ../llamacpp): streaming spill, external nomic catalog, w-dynamics, q8_0 KV.
//
//   kvmem ingest <store_dir> <text_file>   # stream file through a small KV buffer, spill
//                                          # every segment + catalog + source text to disk
//   kvmem ask    <store_dir> "question"    # fresh process: probe catalog -> restore only the
//                                          # relevant segments -> generate the answer
//   kvmem stats  <store_dir>               # manifest + hottest segments by w
//
// Store layout (no JSON libs — line/TSV formats):
//   manifest.txt    key=value: model_file, model_bytes, kv_type, total_tokens, n_segments, build
//   catalog.tsv     id  lo  hi  w  text(escaped)      — text kept next to KV (re-decode insurance)
//   embs.f32        raw float32 [n_segments][EMB_DIM] — nomic embeddings
//   segments/N.kv   serialized KV range (llama_state_seq format; build-sensitive, hence manifest)
//
// Env: MODEL (gguf path), NCTX=4096, CHUNK=512, WIN=256, GEN=96, KVT=q8_0,
//      SEL=3 (max segments), GAP=0.04 (confidence window below top-1), THRESH=0.50 (low-confidence warn)
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>

static const int EMB_DIM = 768; // nomic-embed-text

// ---------- small utils ----------
static std::string env_s(const char *k, const char *d) { const char *v = getenv(k); return v ? v : d; }
static int    env_i(const char *k, int d)    { const char *v = getenv(k); return v ? atoi(v) : d; }
static double env_f(const char *k, double d) { const char *v = getenv(k); return v ? atof(v) : d; }
static void fail(const std::string &m) { fprintf(stderr, "kvmem: %s\n", m.c_str()); exit(1); }

static std::string read_file(const std::string &p) {
    FILE *f = fopen(p.c_str(), "rb"); if (!f) fail("cannot open " + p);
    std::string s; char buf[1 << 16]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f); return s;
}
static void write_file(const std::string &p, const void *d, size_t n, const char *mode = "wb") {
    FILE *f = fopen(p.c_str(), mode); if (!f) fail("cannot write " + p);
    if (n && fwrite(d, 1, n, f) != n) fail("short write " + p);
    fclose(f);
}
static std::string tsv_escape(const std::string &s) {
    std::string o; for (char c : s) { if (c=='\\') o+="\\\\"; else if (c=='\t') o+="\\t"; else if (c=='\n') o+="\\n"; else if (c=='\r') continue; else o+=c; } return o;
}
static std::string tsv_unescape(const std::string &s) {
    std::string o; for (size_t i=0;i<s.size();i++){ if(s[i]=='\\'&&i+1<s.size()){ char c=s[++i]; o += c=='t'?'\t':c=='n'?'\n':c; } else o+=s[i]; } return o;
}

// ---------- store ----------
struct Manifest { std::string model_file, kv_type, build; long model_bytes=0, total_tokens=0, n_segments=0; };
struct CatRow { int id, lo, hi; float w; std::string text; };

static std::string mpath(const std::string &d){return d+"/manifest.txt";}
static bool load_manifest(const std::string &dir, Manifest &m) {
    FILE *f = fopen(mpath(dir).c_str(), "r"); if (!f) return false;
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        std::string s(line); size_t eq = s.find('='); if (eq == std::string::npos) continue;
        std::string k = s.substr(0, eq), v = s.substr(eq + 1); while (!v.empty() && (v.back()=='\n'||v.back()=='\r')) v.pop_back();
        if (k=="model_file") m.model_file=v; else if (k=="kv_type") m.kv_type=v; else if (k=="build") m.build=v;
        else if (k=="model_bytes") m.model_bytes=atol(v.c_str()); else if (k=="total_tokens") m.total_tokens=atol(v.c_str());
        else if (k=="n_segments") m.n_segments=atol(v.c_str());
    }
    fclose(f); return true;
}
static void save_manifest(const std::string &dir, const Manifest &m) {
    char buf[1024];
    int n = snprintf(buf, sizeof buf, "model_file=%s\nmodel_bytes=%ld\nkv_type=%s\nbuild=%s\ntotal_tokens=%ld\nn_segments=%ld\n",
                     m.model_file.c_str(), m.model_bytes, m.kv_type.c_str(), m.build.c_str(), m.total_tokens, m.n_segments);
    write_file(mpath(dir), buf, n);
}
static std::vector<CatRow> load_catalog(const std::string &dir) {
    std::vector<CatRow> rows; FILE *f = fopen((dir+"/catalog.tsv").c_str(), "r"); if (!f) return rows;
    char *line = nullptr; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, f)) > 0) {
        CatRow r; char txt[1 << 15] = {0};
        if (sscanf(line, "%d\t%d\t%d\t%f\t%32766[^\n]", &r.id, &r.lo, &r.hi, &r.w, txt) >= 4) { r.text = tsv_unescape(txt); rows.push_back(r); }
    }
    free(line); fclose(f); return rows;
}
static void save_catalog(const std::string &dir, const std::vector<CatRow> &rows) {
    std::string tmp = dir + "/catalog.tsv.tmp"; FILE *f = fopen(tmp.c_str(), "w"); if (!f) fail("cannot write catalog");
    for (auto &r : rows) fprintf(f, "%d\t%d\t%d\t%.5f\t%s\n", r.id, r.lo, r.hi, r.w, tsv_escape(r.text).c_str());
    fclose(f); rename(tmp.c_str(), (dir + "/catalog.tsv").c_str());
}

// ---------- embedder (ollama nomic) ----------
static std::string utf8_sanitize(const std::string &s) { // drop invalid sequences (byte-cut files etc.)
    std::string o; size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = s[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!len || i + len > n) { i++; continue; }
        bool ok = true;
        for (int k = 1; k < len; k++) if ((s[i+k] & 0xC0) != 0x80) { ok = false; break; }
        if (ok) { o.append(s, i, len); i += len; } else i++;
    }
    return o;
}
static std::string json_escape(const std::string &s) { // full escaping — real text has quotes/tabs/controls
    std::string o; char hex[8];
    for (unsigned char c : s) {
        if (c=='"') o += "\\\""; else if (c=='\\') o += "\\\\";
        else if (c=='\n') o += "\\n"; else if (c=='\t') o += "\\t"; else if (c=='\r') o += "\\r";
        else if (c < 0x20) { snprintf(hex, 8, "\\u%04x", c); o += hex; }
        else o += (char)c;
    }
    return o;
}
static std::vector<float> embed(const std::string &text, bool query) {
    static std::map<std::string, std::vector<float>> cache;
    auto key = (query ? "q:" : "d:") + text;
    auto it = cache.find(key); if (it != cache.end()) return it->second;
    // payload goes through a temp file — never through shell quoting (real text breaks '...')
    std::string clean = utf8_sanitize(text);
    if (clean.size() > 6000) clean = utf8_sanitize(clean.substr(0, 6000)); // belt: nomic ctx is finite
    std::string payload = std::string("{\"model\":\"nomic-embed-text\",\"prompt\":\"")
        + (query ? "search_query: " : "search_document: ") + json_escape(clean) + "\"}";
    char tmpl[] = "/tmp/kvmem_emb_XXXXXX"; int fd = mkstemp(tmpl);
    if (fd < 0) fail("mkstemp failed");
    if (write(fd, payload.data(), payload.size()) != (ssize_t)payload.size()) fail("tmp write failed");
    close(fd);
    std::string cmd = std::string("curl -s --max-time 60 localhost:11434/api/embeddings -d @") + tmpl;
    FILE *p = popen(cmd.c_str(), "r"); std::string resp; char buf[8192]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, p)) > 0) resp.append(buf, n);
    int rc = pclose(p);
    if (!strstr(resp.c_str(), "\"embedding\":[")) {
        rename(tmpl, "/tmp/kvmem_emb_fail.json");
        fprintf(stderr, "[embed] curl rc=%d payload_bytes=%zu (kept /tmp/kvmem_emb_fail.json)\n", rc, payload.size());
    } else unlink(tmpl);
    std::vector<float> v; const char *s = strstr(resp.c_str(), "\"embedding\":[");
    if (!s) fail("embedding failed (server said: " + resp.substr(0, 160) + ").\nkvmem needs ollama with nomic-embed-text:\n  curl -fsSL https://ollama.com/install.sh | sh && ollama pull nomic-embed-text");
    s += 13; char *e;
    while (*s && *s != ']') { v.push_back((float)strtod(s, &e)); s = (*e==',') ? e+1 : e; }
    if ((int)v.size() != EMB_DIM) fail("unexpected embedding size");
    cache[key] = v; return v;
}
static float cosine(const float *a, const float *b) {
    double d=0,na=0,nb=0; for (int i=0;i<EMB_DIM;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];} return (float)(d/(sqrt(na)*sqrt(nb)+1e-9));
}

// ---------- llama ----------
static llama_context *g_ctx; static llama_model *g_model; static const llama_vocab *g_vocab;
static void load_model(const std::string &model_path, int nctx, int nbatch, const std::string &kvt) {
    struct stat st; if (stat(model_path.c_str(), &st)) fail("model file not found: " + model_path + " (set MODEL=...)");
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = env_i("NGL", 99);
    g_model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!g_model) fail("model load failed (GPU memory busy? stop other GPU services or set NGL=0 for CPU)");
    g_vocab = llama_model_get_vocab(g_model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nctx; cp.n_batch = nbatch; cp.n_seq_max = 4;
    cp.kv_unified = true;                                  // seq_cp needs the full KV buffer (poc18a)
    if (kvt != "f16") { cp.type_k = cp.type_v = (kvt=="q8_0") ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; }
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) fail("context init failed (GPU memory busy?)");
}
static std::vector<llama_token> tokenize(const std::string &t, bool add_special) {
    int n = -llama_tokenize(g_vocab, t.c_str(), (int)t.size(), nullptr, 0, add_special, false);
    std::vector<llama_token> out(std::max(n, 0));
    if (n > 0) llama_tokenize(g_vocab, t.c_str(), (int)t.size(), out.data(), n, add_special, false);
    return out;
}
static float *decode(const llama_token *toks, int n, llama_pos pos0, bool want_logits) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[b.n_tokens]=toks[i]; b.pos[b.n_tokens]=pos0+i; b.n_seq_id[b.n_tokens]=1; b.seq_id[b.n_tokens][0]=0;
        b.logits[b.n_tokens]=(want_logits && i==n-1)?1:0; b.n_tokens++;
    }
    int drc = llama_decode(g_ctx, b);
    if (drc) fail("decode failed rc=" + std::to_string(drc) + " (n_tokens=" + std::to_string(n) + " pos0=" + std::to_string((long)pos0) + ")");
    float *lg = want_logits ? llama_get_logits_ith(g_ctx, b.n_tokens-1) : nullptr;
    llama_batch_free(b); return lg;
}

// ---------- segmentation: paragraphs -> sentences/lines -> hard split, capped by tokens ----------
static void hard_split(const std::string &u, int cap, std::vector<std::string> &out) {
    if ((int)tokenize(u, false).size() <= cap || u.size() < 8) { out.push_back(u); return; }
    size_t mid = u.size() / 2;                       // halve at a space if possible
    size_t sp = u.find(' ', mid); if (sp == std::string::npos || sp > u.size() - 4) sp = mid;
    hard_split(u.substr(0, sp), cap, out); hard_split(u.substr(sp), cap, out);
}
static std::vector<std::string> segment_text(const std::string &raw, int cap_tokens) {
    std::vector<std::string> paras; std::string cur;
    for (size_t i = 0; i < raw.size(); i++) {
        cur += raw[i];
        if (raw[i]=='\n' && i+1<raw.size() && raw[i+1]=='\n') { paras.push_back(cur); while (i+1<raw.size() && raw[i+1]=='\n') i++; cur.clear(); }
    }
    if (!cur.empty()) paras.push_back(cur);
    std::vector<std::string> segs;
    for (auto &p : paras) {
        if (p.find_first_not_of(" \t\n\r") == std::string::npos) continue;
        if ((int)tokenize(" "+p, false).size() <= cap_tokens) { segs.push_back(" "+p); continue; }
        // units = pieces ending at a sentence boundary OR a newline (lists/code have no periods)
        std::vector<std::string> units; size_t start = 0;
        for (size_t i = 0; i < p.size(); i++) {
            bool sent = (p[i]=='.'||p[i]=='!'||p[i]=='?') && (i+1>=p.size() || p[i+1]==' ' || p[i+1]=='\n');
            if (sent || p[i]=='\n' || i+1 == p.size()) { units.push_back(p.substr(start, i-start+1)); start = i+1; }
        }
        std::vector<std::string> safe;                // any oversized unit gets hard-split
        for (auto &u : units) hard_split(u, cap_tokens, safe);
        std::string acc;                              // greedy pack units up to the cap
        for (auto &u : safe) {
            if (!acc.empty() && (int)tokenize(" "+acc+u, false).size() > cap_tokens) { segs.push_back(" "+acc); acc.clear(); }
            acc += u;
        }
        if (!acc.empty()) segs.push_back(" "+acc);
    }
    return segs;
}

// ---------- spill one token range to a host buffer via temp seq (poc18a primitive) ----------
static std::vector<uint8_t> spill_range(llama_memory_t mem, int lo, int hi) {
    llama_memory_seq_cp(mem, 0, 1, lo, hi);
    size_t sz = llama_state_seq_get_size(g_ctx, 1);
    std::vector<uint8_t> buf(sz);
    if (llama_state_seq_get_data(g_ctx, buf.data(), sz, 1) != sz) fail("state_seq_get_data failed");
    llama_memory_seq_rm(mem, 1, -1, -1);
    return buf;
}
static void restore_buf(llama_memory_t mem, const std::vector<uint8_t> &buf) {
    if (!llama_state_seq_set_data(g_ctx, buf.data(), buf.size(), 1)) fail("state_seq_set_data failed (store from a different llama.cpp build/model? see manifest.txt)");
    llama_memory_seq_cp(mem, 1, 0, -1, -1);
    llama_memory_seq_rm(mem, 1, -1, -1);
}

// ---------- commands ----------
static int cmd_ingest(const std::string &dir, const std::string &file) {
    const std::string model = env_s("MODEL", "models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf");
    const std::string kvt = env_s("KVT", "q8_0");
    const int NCTX=env_i("NCTX",4096), CHUNK=env_i("CHUNK",512), WIN=env_i("WIN",256), CAP=env_i("SEGCAP",200);
    struct stat st;
    mkdir(dir.c_str(), 0755); mkdir((dir+"/segments").c_str(), 0755);
    Manifest m; bool existed = load_manifest(dir, m);
    if (stat(model.c_str(), &st)) fail("model not found: " + model);
    std::string mbase = model.substr(model.find_last_of('/')+1);
    if (existed && (m.model_file != mbase || m.model_bytes != (long)st.st_size))
        fail("store was built with model '"+m.model_file+"' — refusing to mix models in one store");
    if (existed && m.kv_type != kvt) fail("store kv_type="+m.kv_type+" but KVT="+kvt);
    if (!existed) { m.model_file=mbase; m.model_bytes=st.st_size; m.kv_type=kvt; m.build=env_s("KVMEM_BUILD","b9297"); }

    load_model(model, NCTX, CHUNK, kvt);
    llama_memory_t mem = llama_get_memory(g_ctx);
    std::string raw = read_file(file);
    auto seg_texts = segment_text(raw, CAP);
    if (seg_texts.empty()) fail("no text segments found in " + file);
    fprintf(stderr, "[ingest] %s: %zu segments, base pos %ld\n", file.c_str(), seg_texts.size(), m.total_tokens);

    // token stream + absolute spans
    std::vector<llama_token> ids; std::vector<CatRow> rows;
    for (size_t s = 0; s < seg_texts.size(); s++) {
        auto tk = tokenize(seg_texts[s], m.total_tokens==0 && s==0);
        CatRow r; r.id = (int)(m.n_segments + s); r.lo = (int)(m.total_tokens + ids.size());
        ids.insert(ids.end(), tk.begin(), tk.end());
        r.hi = (int)(m.total_tokens + ids.size()); r.w = 0.f; r.text = seg_texts[s];
        rows.push_back(r);
    }
    const int base = (int)m.total_tokens, N = (int)ids.size();

    // streaming prefill with spill (poc18c), EOF flushes everything incl. the tail window
    auto t0 = std::chrono::steady_clock::now();
    size_t next_spill = 0; int kept_from = base;
    auto spill_upto = [&](int cutoff) {
        while (next_spill < rows.size() && rows[next_spill].hi <= cutoff) {
            auto buf = spill_range(mem, rows[next_spill].lo, rows[next_spill].hi);
            write_file(dir+"/segments/"+std::to_string(rows[next_spill].id)+".kv", buf.data(), buf.size());
            next_spill++;
        }
        int rm_to = next_spill < rows.size() ? std::min(cutoff, rows[next_spill].lo) : cutoff;
        if (rm_to > kept_from) { llama_memory_seq_rm(mem, 0, kept_from, rm_to); kept_from = rm_to; }
    };
    for (int i = 0; i < N; i += CHUNK) {
        int e = std::min(i + CHUNK, N);
        decode(ids.data() + i, e - i, base + i, false);
        spill_upto(base + e - WIN);
    }
    spill_upto(base + N);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    llama_free(g_ctx); llama_model_free(g_model); g_ctx = nullptr; // free VRAM BEFORE embeddings —
    // ollama needs GPU room to load nomic; with the 7B resident it gets CUDA-OOM and returns 500

    // catalog embeddings (memoized per unique text)
    std::string embs; embs.reserve(rows.size()*EMB_DIM*4);
    for (auto &r : rows) { auto e = embed(r.text, false); embs.append((const char*)e.data(), EMB_DIM*4); }
    write_file(dir+"/embs.f32", embs.data(), embs.size(), existed ? "ab" : "wb");
    { FILE *f = fopen((dir+"/catalog.tsv").c_str(), existed ? "a" : "w"); if (!f) fail("catalog write");
      for (auto &r : rows) fprintf(f, "%d\t%d\t%d\t%.5f\t%s\n", r.id, r.lo, r.hi, r.w, tsv_escape(r.text).c_str());
      fclose(f); }
    m.total_tokens = base + N; m.n_segments += (long)rows.size();
    save_manifest(dir, m);
    fprintf(stderr, "[ingest] done: +%d tokens (%.0f tok/s) -> total %ld tokens, %ld segments\n", N, N/secs, m.total_tokens, m.n_segments);
    return 0;
}

static int cmd_ask(const std::string &dir, const std::string &question) {
    Manifest m; if (!load_manifest(dir, m)) fail("no store at " + dir + " (run: kvmem ingest <dir> <file>)");
    const std::string model = env_s("MODEL", "models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf");
    struct stat st; if (stat(model.c_str(), &st)) fail("model not found: " + model);
    std::string mbase = model.substr(model.find_last_of('/')+1);
    if (m.model_file != mbase || m.model_bytes != (long)st.st_size)
        fail("store was built with model '"+m.model_file+"' ("+std::to_string(m.model_bytes)+" bytes), current MODEL is '"+mbase+"' — KV is model-locked. Re-ingest the source text with the new model.");
    auto rows = load_catalog(dir);
    if ((long)rows.size() != m.n_segments) fail("catalog/manifest mismatch");
    std::string embraw = read_file(dir+"/embs.f32");
    if (embraw.size() != rows.size()*EMB_DIM*4) fail("embs.f32 size mismatch");
    const float *E = (const float*)embraw.data();

    const int SEL=env_i("SEL",3), GEN=env_i("GEN",96); const double GAP=env_f("GAP",0.04), THRESH=env_f("THRESH",0.65), DECAY=env_f("DECAY",0.9);
    auto q = embed(question, true);
    std::vector<std::pair<float,int>> scored;
    for (size_t i = 0; i < rows.size(); i++) scored.push_back({cosine(q.data(), E + i*EMB_DIM), (int)i});
    std::sort(scored.rbegin(), scored.rend());
    // confidence-aware selection (poc18f lesson): top-1 always; more only if within GAP of top-1
    std::vector<int> pick = {scored[0].second};
    for (int k = 1; k < (int)scored.size() && (int)pick.size() < SEL; k++)
        if (scored[k].first >= scored[0].first - GAP) pick.push_back(scored[k].second);
    bool low_conf = scored[0].first < THRESH;
    fprintf(stderr, "[ask] probe top: "); for (int k=0;k<3 && k<(int)scored.size();k++) fprintf(stderr, "seg%d(%.3f) ", scored[k].second, scored[k].first);
    fprintf(stderr, "| picked %zu seg(s)%s\n", pick.size(), low_conf ? "  ⚠ LOW CONFIDENCE — likely not in memory" : "");

    load_model(model, env_i("NCTX",4096), env_i("CHUNK",512), m.kv_type);
    llama_memory_t mem = llama_get_memory(g_ctx);
    // true attention sink: a BOS token at pos 0 (structural, no content — content-sinks contaminate the answer)
    int maxhi = 0;
    { llama_token bos = llama_vocab_bos(g_vocab);
      if (bos >= 0) decode(&bos, 1, 0, false); }
    for (int id : pick) {
        std::string p = dir+"/segments/"+std::to_string(rows[id].id)+".kv";
        std::string b = read_file(p); std::vector<uint8_t> buf(b.begin(), b.end());
        restore_buf(mem, buf);
        maxhi = std::max(maxhi, rows[id].hi);
    }
    // answer just after the furthest restored segment (a nearby KV cell must exist; pos is relative for RoPE)
    std::string prompt = "\n Question: " + question + "\n Answer:";
    auto qids = tokenize(prompt, false);
    float *lg = decode(qids.data(), (int)qids.size(), (llama_pos)maxhi, true);
    const int nv = llama_vocab_n_tokens(g_vocab);
    llama_token eos = llama_vocab_eos(g_vocab);
    std::string out; llama_pos pos = (llama_pos)maxhi + (llama_pos)qids.size();
    for (int g = 0; g < GEN; g++) {
        llama_token best = 0; float bv = lg[0];
        for (int i = 1; i < nv; i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
        if (best == eos) break;
        char piece[128]; int pn = llama_token_to_piece(g_vocab, best, piece, sizeof piece, 0, false);
        if (pn > 0) out.append(piece, pn);
        if (out.size() > 2 && out.find("\n\n") != std::string::npos) break;
        lg = decode(&best, 1, pos++, true);
    }
    printf("%s\n", out.c_str());
    // living update: decay all, recharge each genuinely picked segment by its own probe score
    std::map<int,float> score_of; for (auto &si : scored) score_of[si.second] = si.first;
    for (auto &r : rows) r.w *= (float)DECAY;
    for (int id : pick) rows[id].w += score_of[id]; // recharge each genuinely picked segment
    save_catalog(dir, rows);
    return low_conf ? 2 : 0;
}

static int cmd_stats(const std::string &dir) {
    Manifest m; if (!load_manifest(dir, m)) fail("no store at " + dir);
    printf("model=%s (%ld bytes)  kv=%s  build=%s\ntokens=%ld  segments=%ld\n",
           m.model_file.c_str(), m.model_bytes, m.kv_type.c_str(), m.build.c_str(), m.total_tokens, m.n_segments);
    auto rows = load_catalog(dir);
    std::sort(rows.begin(), rows.end(), [](const CatRow&a, const CatRow&b){return a.w>b.w;});
    printf("hottest segments (w):\n");
    for (int i = 0; i < 5 && i < (int)rows.size(); i++) {
        std::string t = rows[i].text.substr(0, 70);
        printf("  seg%-5d w=%.3f  %s...\n", rows[i].id, rows[i].w, t.c_str());
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "kvmem — living-KV session memory (ingest once, ask forever)\n"
                        "  kvmem ingest <store_dir> <text_file>\n"
                        "  kvmem ask    <store_dir> \"question\"\n"
                        "  kvmem stats  <store_dir>\n"
                        "env: MODEL=<gguf> KVT=q8_0 NCTX=4096 SEL=3 GAP=0.04 THRESH=0.50 GEN=96\n");
        return 1;
    }
    llama_log_set([](ggml_log_level l, const char *t, void*){ if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    std::string cmd = argv[1], dir = argv[2];
    if (cmd == "ingest" && argc > 3) return cmd_ingest(dir, argv[3]);
    if (cmd == "ask"    && argc > 3) return cmd_ask(dir, argv[3]);
    if (cmd == "stats")              return cmd_stats(dir);
    fprintf(stderr, "kvmem: bad arguments\n"); return 1;
}
// compile:
// g++ -O2 -o kvmem kvmem.cpp -I <llama.cpp>/include -I <llama.cpp>/ggml/include \
//     -L <llama.cpp>/build/bin -lllama -Wl,-rpath,<llama.cpp>/build/bin
