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
//   kvmem chat   <store_dir>               # interactive REPL: every turn is appended to the
//                                          # store; a later chat process remembers this one
//   kvmem defrag <store_dir>               # rebuild store from the stored text (compact
//                                          # positions; set MODEL=... to migrate to a new model)
//   kvmem stats  <store_dir>               # manifest + hottest segments by w
//
// Store layout (no JSON libs — line/TSV formats):
//   manifest.txt    key=value: model_file, model_bytes, kv_type, total_tokens, n_segments, build
//   catalog.tsv     id  lo  hi  w  text(escaped)      — text kept next to KV (re-decode insurance)
//   embs.f32        raw float32 [n_segments][EMB_DIM] — nomic embeddings
//   segments/N.kv   serialized KV range (llama_state_seq format; build-sensitive, hence manifest)
//
// Env: MODEL (gguf path), NCTX=4096, CHUNK=512, WIN=256, GEN=96, KVT=q8_0,
//      SEL=3 (max segments), GAP=0.04 (confidence window below top-1), THRESH=0.65 (low-confidence warn),
//      COPYB=2.0 (copy-bias: logit bonus for tokens present in the recalled segments' stored text)
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cctype>
#include <sys/socket.h>
#include <netinet/in.h>
#include <csignal>
#include <cerrno>

// Embedder is a per-store property (M6): mixing embedding models in one embs.f32 is garbage,
// so the model name + dim live in the manifest and are applied on store open. Legacy stores
// (no emb_* keys) default to nomic/768. `kvmem reembed <dir>` migrates a store (EMB=bge-m3
// fixes the RU channel: nomic is EN-centric, proven on the session-search pilot 2026-06-02).
static int EMB_DIM = 768;                              // 0 = learn from the first embedding
static std::string EMB_MODEL = "nomic-embed-text";

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
struct Manifest { std::string model_file, kv_type, build, emb_model="nomic-embed-text"; long model_bytes=0, total_tokens=0, n_segments=0, emb_dim=768; };
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
        else if (k=="emb_model") m.emb_model=v; else if (k=="emb_dim") m.emb_dim=atol(v.c_str());
    }
    fclose(f); return true;
}
// store's embedder becomes the process embedder; EMB env may only confirm, never silently mix
static void apply_emb(const Manifest &m) {
    EMB_MODEL = m.emb_model; EMB_DIM = (int)m.emb_dim;
    std::string e = env_s("EMB", "");
    if (!e.empty() && e != EMB_MODEL)
        fail("store is embedded with '" + EMB_MODEL + "' but EMB=" + e + " — run `kvmem reembed` to migrate");
}
static void save_manifest(const std::string &dir, const Manifest &m) {
    char buf[1024];
    int n = snprintf(buf, sizeof buf, "model_file=%s\nmodel_bytes=%ld\nkv_type=%s\nbuild=%s\ntotal_tokens=%ld\nn_segments=%ld\nemb_model=%s\nemb_dim=%ld\n",
                     m.model_file.c_str(), m.model_bytes, m.kv_type.c_str(), m.build.c_str(), m.total_tokens, m.n_segments,
                     m.emb_model.c_str(), m.emb_dim);
    write_file(mpath(dir), buf, n);
}
static std::vector<CatRow> load_catalog(const std::string &dir) {
    // manual tab-split — sscanf's literal '\t' eats ANY whitespace and silently strips the
    // text's leading space, drifting stored text away from what the KV was built on
    std::vector<CatRow> rows; FILE *f = fopen((dir+"/catalog.tsv").c_str(), "r"); if (!f) return rows;
    char *line = nullptr; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, f)) > 0) {
        std::string s(line, len); while (!s.empty() && (s.back()=='\n'||s.back()=='\r')) s.pop_back();
        size_t p = 0, f4[4];
        bool ok = true;
        for (int k = 0; k < 4; k++) { f4[k] = s.find('\t', p); if (f4[k] == std::string::npos) { ok = false; break; } p = f4[k]+1; }
        if (!ok) continue;
        CatRow r;
        r.id = atoi(s.c_str());            r.lo = atoi(s.c_str()+f4[0]+1);
        r.hi = atoi(s.c_str()+f4[1]+1);    r.w  = (float)atof(s.c_str()+f4[2]+1);
        r.text = tsv_unescape(s.substr(f4[3]+1));
        rows.push_back(r);
    }
    free(line); fclose(f); return rows;
}
static void save_catalog(const std::string &dir, const std::vector<CatRow> &rows) {
    // tmp+rename is atomic against crashes but NOT against ENOSPC: an unchecked short write
    // followed by rename replaces a good catalog with a truncated one (lost the cc catalog
    // to a full disk on 2026-06-06). Verify every write before renaming over the old file.
    std::string tmp = dir + "/catalog.tsv.tmp"; FILE *f = fopen(tmp.c_str(), "w"); if (!f) fail("cannot write catalog");
    bool bad = false;
    for (auto &r : rows)
        if (fprintf(f, "%d\t%d\t%d\t%.5f\t%s\n", r.id, r.lo, r.hi, r.w, tsv_escape(r.text).c_str()) < 0) { bad = true; break; }
    if (fflush(f) != 0 || ferror(f)) bad = true;
    if (fclose(f) != 0) bad = true;
    if (bad) { remove(tmp.c_str()); fail("catalog save failed (disk full?) — old catalog kept"); }
    if (rename(tmp.c_str(), (dir + "/catalog.tsv").c_str()) != 0) fail("catalog rename failed — old catalog kept");
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
static bool g_emb_cpu = false; // chat keeps the 7B resident -> force nomic onto CPU (274MB, ms-fast)
static std::vector<float> embed(const std::string &text, bool query) {
    static std::map<std::string, std::vector<float>> cache;
    auto key = (query ? "q:" : "d:") + text;
    auto it = cache.find(key); if (it != cache.end()) return it->second;
    // payload goes through a temp file — never through shell quoting (real text breaks '...')
    std::string clean = utf8_sanitize(text);
    if (clean.size() > 6000) clean = utf8_sanitize(clean.substr(0, 6000)); // belt: embedder ctx is finite
    // task prefixes are a nomic convention; bge-m3 (and most others) are trained without them
    bool nomic = EMB_MODEL.rfind("nomic", 0) == 0;
    std::string payload = std::string("{\"model\":\"") + EMB_MODEL + "\","
        + (g_emb_cpu ? "\"options\":{\"num_gpu\":0}," : "") + "\"prompt\":\""
        + (nomic ? (query ? "search_query: " : "search_document: ") : "") + json_escape(clean) + "\"}";
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
    if (!s) fail("embedding failed (server said: " + resp.substr(0, 160) + ").\nkvmem needs ollama with " + EMB_MODEL + ":\n  curl -fsSL https://ollama.com/install.sh | sh && ollama pull " + EMB_MODEL);
    s += 13; char *e;
    while (*s && *s != ']') { v.push_back((float)strtod(s, &e)); s = (*e==',') ? e+1 : e; }
    if (EMB_DIM == 0) EMB_DIM = (int)v.size();         // new store / reembed: learn the dim
    if ((int)v.size() != EMB_DIM) fail("embedding size " + std::to_string(v.size()) + " != store emb_dim " + std::to_string(EMB_DIM));
    cache[key] = v; return v;
}
static float cosine(const float *a, const float *b) {
    double d=0,na=0,nb=0; for (int i=0;i<EMB_DIM;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];} return (float)(d/(sqrt(na)*sqrt(nb)+1e-9));
}

// ---------- lexical channel (M5): exact-term matching, complements nomic's topical semantics ----------
// nomic ranks topical neighbours for rare exact terms (M4 ISC-4: "kvmem" -> a topical 0.66 miss).
// A word is a run of ASCII-alnum (lowercased) or high UTF-8 bytes (cyrillic kept as-is, no case-fold
// — fine for rare terms); length >= 3. Mirrors wikiq's tokenizer/IDF/RRF.
static std::set<std::string> words_of(const std::string &s) {
    std::set<std::string> out; std::string cur;
    auto flush = [&]{ if (cur.size() >= 3) out.insert(cur); cur.clear(); };
    for (unsigned char c : s) {
        if (c >= 0x80) cur += (char)c;
        else if (c >= 'a' && c <= 'z') cur += (char)c;
        else if (c >= 'A' && c <= 'Z') cur += (char)(c + 32);
        else if (c >= '0' && c <= '9') cur += (char)c;
        else flush();
    }
    flush();
    return out;
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
// tokenizer without weights: for TEXTONLY ingest (catalog+embs only, no KV) the model is needed
// solely for segmentation token counts — vocab_only loads in ms, no VRAM. g_ctx stays null.
static void load_vocab_only(const std::string &model_path) {
    struct stat st; if (stat(model_path.c_str(), &st)) fail("model file not found: " + model_path + " (set MODEL=...)");
    llama_model_params mp = llama_model_default_params(); mp.vocab_only = true;
    g_model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!g_model) fail("vocab load failed: " + model_path);
    g_vocab = llama_model_get_vocab(g_model);
}
static std::vector<llama_token> tokenize(const std::string &t, bool add_special, bool parse_special = false) {
    int n = -llama_tokenize(g_vocab, t.c_str(), (int)t.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> out(std::max(n, 0));
    if (n > 0) llama_tokenize(g_vocab, t.c_str(), (int)t.size(), out.data(), n, add_special, parse_special);
    return out;
}
static float *decode_seq(const llama_token *toks, int n, llama_pos pos0, bool want_logits, int seq) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[b.n_tokens]=toks[i]; b.pos[b.n_tokens]=pos0+i; b.n_seq_id[b.n_tokens]=1; b.seq_id[b.n_tokens][0]=seq;
        b.logits[b.n_tokens]=(want_logits && i==n-1)?1:0; b.n_tokens++;
    }
    int drc = llama_decode(g_ctx, b);
    if (drc) fail("decode failed rc=" + std::to_string(drc) + " (n_tokens=" + std::to_string(n) + " pos0=" + std::to_string((long)pos0) + ")");
    float *lg = want_logits ? llama_get_logits_ith(g_ctx, b.n_tokens-1) : nullptr;
    llama_batch_free(b); return lg;
}
static float *decode(const llama_token *toks, int n, llama_pos pos0, bool want_logits) {
    return decode_seq(toks, n, pos0, want_logits, 0);
}
// greedy generation with copy-bias + stop conditions, in sequence `seq`. `lg0` = logits for the
// first step (already decoded). Returns the generated text. Used by chat's ask-fallback path.
static std::string gen_loop(float *lg, llama_pos pos, int seq, int GEN,
                            const std::vector<uint8_t> &in_copy, float COPYB, bool stop_blank = true) {
    // stop_blank: chat-style answers end at the first blank line; GROUND answers are
    // multi-paragraph (lists etc.) and rely on eos / GEN budget instead.
    const int nv = llama_vocab_n_tokens(g_vocab);
    llama_token eos = llama_vocab_eos(g_vocab);
    std::string out; int ws_run = 0;
    for (int g = 0; g < GEN; g++) {
        llama_token best = 0; float bv = -1e30f;
        for (int i = 0; i < nv; i++) { float v = lg[i] + (!in_copy.empty() && in_copy[i] ? COPYB : 0.f); if (v > bv) { bv = v; best = i; } }
        if (best == eos) break;
        char piece[128]; int pn = llama_token_to_piece(g_vocab, best, piece, sizeof piece, 0, false);
        std::string ps = pn > 0 ? std::string(piece, pn) : "";
        lg = decode_seq(&best, 1, pos++, true, seq);
        out += ps;
        ws_run = (ps.find_first_not_of(" \t\n\r") == std::string::npos) ? ws_run + 1 : 0;
        if (out.size() > 2 && ((stop_blank && out.find("\n\n") != std::string::npos) || ws_run >= 3)) break;
    }
    while (!out.empty() && isspace((unsigned char)out.back())) out.pop_back();
    return out;
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
// restore a saved KV buffer into an arbitrary sequence (chat ask-fallback uses a scratch seq);
// seq 1 stays the transfer temp, as in spill/restore above
static void restore_buf_to(llama_memory_t mem, const std::vector<uint8_t> &buf, int dst) {
    if (!llama_state_seq_set_data(g_ctx, buf.data(), buf.size(), 1)) fail("state_seq_set_data failed");
    llama_memory_seq_cp(mem, 1, dst, -1, -1);
    llama_memory_seq_rm(mem, 1, -1, -1);
}

// ---------- streaming prefill with spill (poc18c) — shared by ingest/defrag/chat ----------
// rows carry absolute [lo,hi) spans; ids is the matching token stream starting at base.
// EOF flushes everything including the tail window. Returns wall seconds.
static double prefill_spill(const std::string &dir, const std::vector<CatRow> &rows,
                            const std::vector<llama_token> &ids, int base, int CHUNK, int WIN) {
    llama_memory_t mem = llama_get_memory(g_ctx);
    const int N = (int)ids.size();
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
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
    if (existed) apply_emb(m);
    else { EMB_MODEL = env_s("EMB", "nomic-embed-text"); EMB_DIM = 0; m.emb_model = EMB_MODEL; }

    // next segment id = max existing id + 1, NOT n_segments — after a prune the ids are sparse
    // and n_segments would collide with (and overwrite) a surviving segment's .kv file
    long base_id = 0;
    { auto old = load_catalog(dir); for (auto &o : old) base_id = std::max(base_id, (long)o.id + 1); }

    // TEXTONLY=1 (post-M5): for log stores the honest product is probe+EXTRACT — the KV is never
    // read (probe = nomic embs + lexical IDF, answer = stored text). Skip prefill entirely:
    // no GPU, no .kv files (21 GB -> ~MB for the cc store), ~1 s/session instead of ~20.
    const bool TEXTONLY = env_i("TEXTONLY", 0) != 0;
    if (TEXTONLY) load_vocab_only(model); else load_model(model, NCTX, CHUNK, kvt);
    std::string raw = read_file(file);
    auto seg_texts = segment_text(raw, CAP);
    if (seg_texts.empty()) fail("no text segments found in " + file);
    fprintf(stderr, "[ingest] %s: %zu segments, base pos %ld\n", file.c_str(), seg_texts.size(), m.total_tokens);

    // token stream + absolute spans
    std::vector<llama_token> ids; std::vector<CatRow> rows;
    for (size_t s = 0; s < seg_texts.size(); s++) {
        auto tk = tokenize(seg_texts[s], m.total_tokens==0 && s==0);
        CatRow r; r.id = (int)(base_id + s); r.lo = (int)(m.total_tokens + ids.size());
        ids.insert(ids.end(), tk.begin(), tk.end());
        r.hi = (int)(m.total_tokens + ids.size()); r.w = 0.f; r.text = seg_texts[s];
        rows.push_back(r);
    }
    const int base = (int)m.total_tokens, N = (int)ids.size();

    double secs = TEXTONLY ? 0.0 : prefill_spill(dir, rows, ids, base, CHUNK, WIN);
    if (g_ctx) llama_free(g_ctx); llama_model_free(g_model); g_ctx = nullptr; // free VRAM BEFORE embeddings —
    // ollama needs GPU room to load nomic; with the 7B resident it gets CUDA-OOM and returns 500

    // catalog embeddings (memoized per unique text)
    std::string embs; embs.reserve(rows.size()*EMB_DIM*4);
    for (auto &r : rows) { auto e = embed(r.text, false); embs.append((const char*)e.data(), EMB_DIM*4); }
    write_file(dir+"/embs.f32", embs.data(), embs.size(), existed ? "ab" : "wb");
    { FILE *f = fopen((dir+"/catalog.tsv").c_str(), existed ? "a" : "w"); if (!f) fail("catalog write");
      for (auto &r : rows) fprintf(f, "%d\t%d\t%d\t%.5f\t%s\n", r.id, r.lo, r.hi, r.w, tsv_escape(r.text).c_str());
      fclose(f); }
    m.total_tokens = base + N; m.n_segments += (long)rows.size(); m.emb_dim = EMB_DIM;
    save_manifest(dir, m);
    fprintf(stderr, "[ingest] done: +%d tokens (%.0f tok/s) -> total %ld tokens, %ld segments\n", N, secs > 0 ? N/secs : 0.0, m.total_tokens, m.n_segments);
    return 0;
}

// hybrid probe (M5): RRF fusion of nomic semantic cosine + lexical IDF-overlap. Catalog-side,
// no model needed for the lexical channel. Returns picked segment ids (document order), the top
// semantic cosine + the top lexical score (for honest low-confidence), and per-segment df cached
// across calls of one resident process. Mirrors wikiq (semantic + IDF, RRF k=60).
static std::vector<int> hybrid_probe(const std::vector<CatRow> &rows, const std::string &embraw,
                                     const std::string &question, int SEL, double GAP, double THRESH,
                                     float &sem_top, float &lex_top, bool &low_conf,
                                     std::map<int,float> &sem_of) {
    const int n = (int)rows.size();
    const float *E = (const float*)embraw.data();
    // per-segment word sets + document frequency (cached on the rows vector identity is overkill;
    // recompute — ~900 segments is microseconds)
    static const std::vector<CatRow> *df_for = nullptr; static std::vector<std::set<std::string>> SEG; static std::map<std::string,int> DF;
    if (df_for != &rows) {
        df_for = &rows; SEG.clear(); DF.clear(); SEG.reserve(n);
        for (auto &r : rows) { SEG.push_back(words_of(r.text)); for (auto &w : SEG.back()) DF[w]++; }
    }
    auto qe = embed(question, true);
    auto qw = words_of(question);
    std::vector<float> sem(n), lex(n);
    for (int i = 0; i < n; i++) {
        sem[i] = cosine(qe.data(), E + i*EMB_DIM);
        // rare exact terms always count (df<=2, e.g. "kvmem"/"cobalt-finch" regardless of corpus
        // size); moderately rare via the df<0.2N stopword cut. Keeps the channel alive on tiny stores.
        double s = 0; for (auto &w : qw) { auto it = DF.find(w); if (it != DF.end() && (it->second <= 2 || it->second < n*0.2)) if (SEG[i].count(w)) s += log((double)n / it->second); }
        lex[i] = (float)s;
    }
    // ranks (0 = best) for each channel
    std::vector<int> order(n); for (int i=0;i<n;i++) order[i]=i;
    std::vector<int> rs(n), rl(n);
    std::sort(order.begin(), order.end(), [&](int a,int b){return sem[a]>sem[b];}); for (int r=0;r<n;r++) rs[order[r]]=r;
    std::sort(order.begin(), order.end(), [&](int a,int b){return lex[a]>lex[b];}); for (int r=0;r<n;r++) rl[order[r]]=r;
    std::vector<std::pair<float,int>> fused;
    for (int i=0;i<n;i++) fused.push_back({1.f/(60+rs[i]) + 1.f/(60+rl[i]), i});
    std::sort(fused.rbegin(), fused.rend());
    sem_top = *std::max_element(sem.begin(), sem.end());
    lex_top = *std::max_element(lex.begin(), lex.end());
    // confidence-aware selection (poc18f): top-1 always; more only if within GAP·top of the fused top
    std::vector<int> pick = {fused[0].second};
    for (int k = 1; k < n && (int)pick.size() < SEL; k++)
        if (fused[k].first >= fused[0].first * (1.0 - GAP*5)) pick.push_back(fused[k].second);
    // low-confidence = semantic UNfamiliarity. The lexical channel decides WHICH segment, not
    // WHETHER we know the topic: a single incidental rare-word match (e.g. "swallow") must NOT
    // signal confidence (M5 verify: it falsely passed an absent fact). Semantic cosine is the
    // honest "is this topic in memory at all" signal.
    low_conf = sem_top < THRESH;
    for (int i=0;i<n;i++) sem_of[i] = sem[i];
    fprintf(stderr, "[probe] sem_top %.3f lex_top %.2f | fused top: ", sem_top, lex_top);
    for (int k=0;k<3 && k<n;k++) fprintf(stderr, "seg%d ", fused[k].second);
    fprintf(stderr, "| picked %zu%s\n", pick.size(), low_conf ? "  ⚠ LOW CONFIDENCE — likely not in memory" : "");
    return pick;
}

// shared by `ask` (one-shot) and `serve` (model stays resident): probe -> restore -> generate.
// Assumes the model is loaded and the memory is empty. Updates w and writes the catalog back.
static std::string ask_core(const std::string &dir, Manifest &m, std::vector<CatRow> &rows,
                            const std::string &embraw, const std::string &question, bool &low_conf) {
    const int SEL=env_i("SEL",3), GEN=env_i("GEN",96); const double GAP=env_f("GAP",0.04), THRESH=env_f("THRESH",0.65), DECAY=env_f("DECAY",0.9);
    float sem_top, lex_top; std::map<int,float> score_of;
    // Text modes (EXTRACT/GROUND) read the full SEL: the GAP confidence-cut is a poc18f lesson
    // about KV-restore generation (similar-but-wrong KV distracts); prompt-grounded reading is
    // the opposite regime — an extra excerpt is cheap and brings the defining segment along
    // (caught: a fresh status segment crowded out the definition, GROUND drifted to its prior).
    const bool textmode = env_i("EXTRACT", 0) || env_i("GROUND", 0);
    std::vector<int> pick = hybrid_probe(rows, embraw, question, SEL, textmode ? 1e9 : GAP, THRESH, sem_top, lex_top, low_conf, score_of);
    std::sort(pick.begin(), pick.end()); // document order reads better than fused order

    // Tiny picked segments are markdown headers/fragments — the probe loves their exact terms
    // ("## Что теперь умеет kvmem") but they carry no content (caught on the full cc store,
    // 38k segments: headers beat bodies). A header is a POINTER to the body right after it, so
    // extend each pick with following document-order neighbours until ~MINCH chars accumulated.
    // Text-reading paths only (EXTRACT/GROUND); the KV-restore path is untouched.
    std::vector<int> read_ids;
    { const int MINCH = env_i("MINCH", 300);
      std::set<int> seen;
      for (int id : pick) {
          int len = 0;
          for (int j = id; j < (int)rows.size() && j <= id + 4; j++) {
              if (len >= MINCH && j > id) break;
              seen.insert(j); len += (int)rows[j].text.size();
          }
      }
      read_ids.assign(seen.begin(), seen.end()); }   // std::set = sorted = document order

    // EXTRACT mode (M5): for messy logs, free generation confabulates (wikiq echo-attractor) —
    // return the stored text of the recalled segments instead. Honest "here's what we had".
    if (env_i("EXTRACT", 0)) {
        std::string out;
        for (int id : read_ids) { out += rows[id].text; if (out.size() && out.back() != '\n') out += '\n'; }
        for (auto &r : rows) r.w *= (float)DECAY; for (int id : pick) rows[id].w += score_of[id];
        save_catalog(dir, rows);
        return out;
    }

    // GROUND mode (M6): probe-picked stored TEXT goes into the prompt as evidence, the model
    // answers in its own words. Free generation from distant KV confabulates on real terms with
    // a competing parametric prior (M5: "kvmem -> virtualized memory" with the right segment
    // restored). Evidence-in-prompt is the regime the model is trained for — prior loses to
    // text under its nose. This is the honest "answer in your own words" for log stores.
    if (env_i("GROUND", 0)) {
        for (auto &r : rows) r.w *= (float)DECAY;
        if (low_conf) { save_catalog(dir, rows); return "(low confidence — likely not in memory)"; }
        for (int id : pick) rows[id].w += score_of[id];
        save_catalog(dir, rows);
        std::string prompt =
            "<|im_start|>system\n"
            "You answer questions from the user's past work-session notes. Use ONLY the excerpts below — "
            "never your own knowledge. If they answer the question, answer it. If they are only related, "
            "summarize what they DO say about the topic and note what is missing. Never guess what a term "
            "means beyond the excerpts. Only if they are entirely unrelated, say \"not in the excerpts\". "
            "Answer in the question's language. Be concise.<|im_end|>\n"
            "<|im_start|>user\nExcerpts:\n";
        for (int id : read_ids) { prompt += "---\n" + rows[id].text + "\n"; }
        prompt += "---\nQuestion: " + question + "<|im_end|>\n<|im_start|>assistant\n";
        auto pids = tokenize(prompt, true, true);          // chat-template tokens are special
        // prefill in n_batch-sized chunks — one llama_decode call is capped by cparams.n_batch
        float *lg = nullptr; const int CH = env_i("CHUNK", 512);
        for (int i = 0; i < (int)pids.size(); i += CH) {
            int e = std::min(i + CH, (int)pids.size());
            lg = decode(pids.data() + i, e - i, i, e == (int)pids.size());
        }
        return gen_loop(lg, (llama_pos)pids.size(), 0, std::max(GEN, 256), {}, 0.f, /*stop_blank=*/false);
    }

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
    // copy-bias (M2): tokens that occur in the recalled segments' STORED TEXT get a logit bonus.
    // Greedy alone paraphrases rare/invented multi-token strings (M1: cobalt-finch -> "backup-server-
    // observatory"); the bonus tips near-ties toward verbatim copy. KV path stays pure — the text is
    // already in the catalog, we only read token IDs from it, never re-prefill it.
    const float COPYB = (float)env_f("COPYB", 2.0);
    std::vector<uint8_t> in_copy;
    if (COPYB > 0) {
        in_copy.assign(nv, 0);
        for (int id : pick) for (auto t : tokenize(rows[id].text, false)) if (t >= 0 && t < nv) in_copy[t] = 1;
    }
    std::string out; llama_pos pos = (llama_pos)maxhi + (llama_pos)qids.size();
    int ws_run = 0; // copy-bias boosts the space token -> "\n \n \n" never matches "\n\n"; count ws-only pieces instead
    for (int g = 0; g < GEN; g++) {
        llama_token best = 0; float bv = -1e30f;
        for (int i = 0; i < nv; i++) {
            float v = lg[i] + (!in_copy.empty() && in_copy[i] ? COPYB : 0.f);
            if (v > bv) { bv = v; best = i; }
        }
        if (best == eos) break;
        char piece[128]; int pn = llama_token_to_piece(g_vocab, best, piece, sizeof piece, 0, false);
        std::string ps = pn > 0 ? std::string(piece, pn) : "";
        ws_run = (ps.find_first_not_of(" \t\n\r") == std::string::npos) ? ws_run + 1 : 0;
        out += ps;
        if (out.size() > 2 && (out.find("\n\n") != std::string::npos || ws_run >= 3)) break;
        lg = decode(&best, 1, pos++, true);
    }
    while (!out.empty() && isspace((unsigned char)out.back())) out.pop_back();
    // living update: decay all, recharge each genuinely picked segment by its semantic probe score
    for (auto &r : rows) r.w *= (float)DECAY;
    for (int id : pick) rows[id].w += score_of[id]; // score_of filled by hybrid_probe (semantic cosine)
    save_catalog(dir, rows);
    return out;
}

// validate store-vs-MODEL and load everything; shared by ask/serve
static void open_store(const std::string &dir, Manifest &m, std::vector<CatRow> &rows, std::string &embraw, std::string &model) {
    if (!load_manifest(dir, m)) fail("no store at " + dir + " (run: kvmem ingest <dir> <file>)");
    apply_emb(m);
    model = env_s("MODEL", "models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf");
    struct stat st; if (stat(model.c_str(), &st)) fail("model not found: " + model);
    std::string mbase = model.substr(model.find_last_of('/')+1);
    if (m.model_file != mbase || m.model_bytes != (long)st.st_size)
        fail("store was built with model '"+m.model_file+"' ("+std::to_string(m.model_bytes)+" bytes), current MODEL is '"+mbase+"' — KV is model-locked. Run `kvmem defrag` with the new MODEL to migrate.");
    rows = load_catalog(dir);
    if ((long)rows.size() != m.n_segments) fail("catalog/manifest mismatch");
    embraw = read_file(dir+"/embs.f32");
    if (embraw.size() != rows.size()*EMB_DIM*4) fail("embs.f32 size mismatch");
}

static int cmd_ask(const std::string &dir, const std::string &question) {
    Manifest m; std::vector<CatRow> rows; std::string embraw, model;
    open_store(dir, m, rows, embraw, model);
    // EXTRACT never touches the model: probe = catalog-side (nomic + lexical IDF), answer =
    // stored text. Skipping the load makes EXTRACT GPU-free and works on TEXTONLY stores.
    if (!env_i("EXTRACT", 0)) load_model(model, env_i("NCTX",4096), env_i("CHUNK",512), m.kv_type);
    bool low_conf = false;
    std::string out = ask_core(dir, m, rows, embraw, question, low_conf);
    printf("%s\n", out.c_str());
    return low_conf ? 2 : 0;
}

// probe (curiosity-loop M5 / drive layer): GPU-free gap SENSOR. Embeds the question (ollama),
// runs the hybrid probe against the catalog, prints machine-readable scores. NO model load
// (works on TEXTONLY stores, no MODEL env needed), NO w update — scanning for holes is not
// usage; the sensor must not recharge what it measures (read-only by design).
// stdout TSV: sem_top  lex_top  low_conf  top_seg_id  top_seg_snippet
// exit 0 = topic known, 2 = low confidence (hole in memory).
static int cmd_probe(const std::string &dir, const std::string &question) {
    Manifest m; if (!load_manifest(dir, m)) fail("no store at " + dir + " (run: kvmem ingest <dir> <file>)");
    apply_emb(m);
    std::vector<CatRow> rows = load_catalog(dir);
    if ((long)rows.size() != m.n_segments) fail("catalog/manifest mismatch");
    std::string embraw = read_file(dir+"/embs.f32");
    if (embraw.size() != rows.size()*EMB_DIM*4) fail("embs.f32 size mismatch");
    const int SEL = env_i("SEL",3);
    const double GAP = env_f("GAP",0.04), THRESH = env_f("THRESH",0.65);
    // batch mode (question == "-"): one question per stdin line, one TSV line each — the store
    // and the DF cache load once, so a 50-question curiosity scan is seconds, not minutes.
    auto probe_one = [&](const std::string &q) {
        float sem_top=0, lex_top=0; bool low_conf=false; std::map<int,float> sem_of;
        std::vector<int> pick = hybrid_probe(rows, embraw, q, SEL, GAP, THRESH,
                                             sem_top, lex_top, low_conf, sem_of);
        std::string snip = utf8_sanitize(rows[pick[0]].text.substr(0, 120)); // substr may cut a multibyte char
        printf("%.4f\t%.2f\t%d\t%d\t%s\n", sem_top, lex_top, low_conf ? 1 : 0, pick[0], tsv_escape(snip).c_str());
        fflush(stdout);
        return low_conf;
    };
    if (question == "-") {
        bool any_low = false; std::string line;
        for (char buf[8192]; fgets(buf, sizeof buf, stdin); ) {
            line = buf; while (!line.empty() && (line.back()=='\n' || line.back()=='\r')) line.pop_back();
            if (!line.empty()) any_low |= probe_one(line);
        }
        return any_low ? 2 : 0;
    }
    return probe_one(question) ? 2 : 0;
}

// chat (M2): interactive REPL with living memory. The model loads ONCE; every turn
// (user line + answer) is committed to the store immediately (KV spill + catalog + embeddings
// + manifest), so a crash loses nothing and a LATER chat/ask process remembers this one.
// Recall: each user line probes the catalog; relevant old segments are restored before
// answering and removed from the live window after (disk keeps them). Startup restores the
// tail segment of the store -> the conversation resumes adjacent to existing KV cells
// (fresh-cache decode far from any cell fails with rc=-1 — M1 lesson #5).
static int cmd_chat(const std::string &dir) {
    const std::string model = env_s("MODEL", "models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf");
    struct stat st; if (stat(model.c_str(), &st)) fail("model not found: " + model);
    std::string mbase = model.substr(model.find_last_of('/')+1);
    const std::string kvt = env_s("KVT", "q8_0");
    const int NCTX=env_i("NCTX",4096), GEN=env_i("GEN",192), SEL=env_i("SEL",3);
    const double GAP=env_f("GAP",0.04), THRESH=env_f("THRESH",0.65), DECAY=env_f("DECAY",0.9);
    const float COPYB = (float)env_f("COPYB", 2.0);
    g_emb_cpu = env_i("EMB_CPU", 1) != 0;

    mkdir(dir.c_str(), 0755); mkdir((dir+"/segments").c_str(), 0755);
    Manifest m; bool existed = load_manifest(dir, m);
    if (existed && (m.model_file != mbase || m.model_bytes != (long)st.st_size))
        fail("store was built with model '"+m.model_file+"' — KV is model-locked (kvmem defrag migrates)");
    if (existed && m.kv_type != kvt) fail("store kv_type="+m.kv_type+" but KVT="+kvt);
    if (!existed) { m.model_file=mbase; m.model_bytes=st.st_size; m.kv_type=kvt; m.build=env_s("KVMEM_BUILD","b9297"); }
    if (existed) apply_emb(m);
    else { EMB_MODEL = env_s("EMB", "nomic-embed-text"); EMB_DIM = 0; m.emb_model = EMB_MODEL; }
    auto rows = load_catalog(dir);
    std::string embraw = existed ? read_file(dir+"/embs.f32") : std::string();
    if (embraw.size() != rows.size()*EMB_DIM*4) fail("embs.f32 size mismatch");

    load_model(model, NCTX, env_i("CHUNK",512), kvt);
    llama_memory_t mem = llama_get_memory(g_ctx);
    const int nv = llama_vocab_n_tokens(g_vocab);
    llama_token eos = llama_vocab_eos(g_vocab);

    // anchor: BOS sink + the tail segment of the store (resume where we left off);
    // a fresh store needs neither — the first turn carries its own BOS at pos 0
    int live_lo = 0;                                   // oldest live cell we still keep in cache
    if (!rows.empty()) {
        llama_token bos = llama_vocab_bos(g_vocab); if (bos >= 0) decode(&bos, 1, 0, false);
        std::string b = read_file(dir+"/segments/"+std::to_string(rows.back().id)+".kv");
        std::vector<uint8_t> buf(b.begin(), b.end());
        restore_buf(mem, buf);
        live_lo = rows.back().lo;
    }
    fprintf(stderr, "[chat] %s: %ld tokens, %zu segments. /quit to exit.\n", dir.c_str(), m.total_tokens, rows.size());

    char *line = nullptr; size_t lcap = 0;
    while (true) {
        fprintf(stderr, "you> "); fflush(stderr);
        ssize_t len = getline(&line, &lcap, stdin);
        if (len <= 0) break;
        std::string q(line, len); while (!q.empty() && (q.back()=='\n'||q.back()=='\r')) q.pop_back();
        if (q.empty()) continue;
        if (q == "/quit" || q == "/exit") break;

        // probe the whole catalog (old sessions + earlier turns of this one) — hybrid (M5)
        std::vector<int> pick; float top = 0.f; std::map<int,float> score_of;
        if (!rows.empty()) {
            float sem_top, lex_top; bool lc;
            auto cand = hybrid_probe(rows, embraw, q, SEL, GAP, THRESH, sem_top, lex_top, lc, score_of);
            top = sem_top;
            // restore confident hits not already live in the window
            if (!lc) for (int id : cand) { if (rows[id].hi > live_lo) continue; pick.push_back(id); }
            for (auto &r : rows) r.w *= (float)DECAY;
            for (int id : pick) rows[id].w += score_of[id];
            fprintf(stderr, "[recall] sem %.3f lex %.2f, restored %zu old seg(s)\n", sem_top, lex_top, pick.size());
        }
        // copy-bias source: recalled segments' stored text (used by both paths)
        std::vector<uint8_t> in_copy;
        if (COPYB > 0 && !pick.empty()) {
            in_copy.assign(nv, 0);
            for (int id : pick) for (auto t : tokenize(rows[id].text, false)) if (t >= 0 && t < nv) in_copy[t] = 1;
        }

        std::string out; int turn_lo = (int)m.total_tokens, turn_hi;
        const bool FALLBACK = env_i("FALLBACK", 1) != 0;
        bool used_fallback = FALLBACK && !pick.empty(); // a confident hit (either channel) was restored

        if (used_fallback) {
            // ASK-FALLBACK (M4): the M3 A/B showed the conversational prior overrides injected KV
            // when answering doc-recall at the live head. So on a confident doc hit, answer in a
            // SCRATCH sequence ask-style (BOS + segment + question, no dialogue tail), then teacher-
            // force the result back into the conversation so continuity + the store stay intact.
            const int SC = 2;
            llama_memory_seq_rm(mem, SC, -1, -1);
            int sp = 0; llama_token bos = llama_vocab_bos(g_vocab);
            if (bos >= 0) { decode_seq(&bos, 1, 0, false, SC); sp = 1; }   // sink at pos 0
            std::vector<int> order = pick;
            std::sort(order.begin(), order.end(), [&](int a, int b){ return rows[a].lo < rows[b].lo; });
            for (int id : order) {                                          // compact segments right after the sink
                std::string b = read_file(dir+"/segments/"+std::to_string(rows[id].id)+".kv");
                std::vector<uint8_t> buf(b.begin(), b.end());
                restore_buf_to(mem, buf, SC);
                if (rows[id].lo != sp) llama_memory_seq_add(mem, SC, rows[id].lo, rows[id].hi, sp - rows[id].lo);
                sp += rows[id].hi - rows[id].lo;
            }
            std::string ap = "\n Question: " + q + "\n Answer:";
            auto aq = tokenize(ap, false);
            float *lg = decode_seq(aq.data(), (int)aq.size(), sp, true, SC);
            out = gen_loop(lg, sp + (int)aq.size(), SC, GEN, in_copy, COPYB);
            llama_memory_seq_rm(mem, SC, -1, -1);
            printf(" %s\n", out.c_str());
            // materialize the turn in the live conversation (seq 0) so the next turn stays coherent
            std::string turn = "\nUser: " + q + "\nAssistant: " + out;
            auto tids = tokenize(turn, m.total_tokens == 0);
            decode(tids.data(), (int)tids.size(), (llama_pos)turn_lo, false);
            turn_hi = turn_lo + (int)tids.size();
        } else {
            // normal conversational turn at the live head; restored segments (if any) are RoPE-
            // shifted next to the conversation (KVSHIFT) and dropped after the answer
            const bool KVSHIFT = env_i("KVSHIFT", 1) != 0;
            std::vector<std::pair<int,int>> restored_spans;
            if (!pick.empty()) {
                std::vector<int> order = pick;
                std::sort(order.begin(), order.end(), [&](int a, int b){ return rows[a].lo < rows[b].lo; });
                int total_len = 0; for (int id : order) total_len += rows[id].hi - rows[id].lo;
                int place = live_lo - total_len; bool can_shift = KVSHIFT && place >= 1;
                for (int id : order) {
                    std::string b = read_file(dir+"/segments/"+std::to_string(rows[id].id)+".kv");
                    std::vector<uint8_t> buf(b.begin(), b.end());
                    restore_buf(mem, buf);
                    int L = rows[id].hi - rows[id].lo;
                    if (can_shift && rows[id].hi != live_lo) { llama_memory_seq_add(mem, 0, rows[id].lo, rows[id].hi, place - rows[id].lo); restored_spans.push_back({place, place + L}); }
                    else restored_spans.push_back({rows[id].lo, rows[id].hi});
                    if (can_shift) place += L;
                }
            }
            std::string turn = "\nUser: " + q + "\nAssistant:";
            auto tids = tokenize(turn, m.total_tokens == 0);
            float *lg = decode(tids.data(), (int)tids.size(), (llama_pos)turn_lo, true);
            llama_pos pos = (llama_pos)turn_lo + (llama_pos)tids.size();
            int ws_run = 0; // invariant: every appended char is a DECODED token (text==KV span)
            for (int g = 0; g < GEN; g++) {
                llama_token best = 0; float bv = -1e30f;
                for (int i = 0; i < nv; i++) { float v = lg[i] + (!in_copy.empty() && in_copy[i] ? COPYB : 0.f); if (v > bv) { bv = v; best = i; } }
                if (best == eos) break;
                char piece[128]; int pn = llama_token_to_piece(g_vocab, best, piece, sizeof piece, 0, false);
                std::string ps = pn > 0 ? std::string(piece, pn) : "";
                if (out.size() > 2 && (out + ps).find("\nUser:") != std::string::npos) break; // rejected, not decoded
                lg = decode(&best, 1, pos++, true);
                out += ps; printf("%s", ps.c_str()); fflush(stdout);
                ws_run = (ps.find_first_not_of(" \t\n\r") == std::string::npos) ? ws_run + 1 : 0;
                if (out.size() > 2 && (out.find("\n\n") != std::string::npos || ws_run >= 3)) break;
            }
            printf("\n");
            for (auto &sp : restored_spans) llama_memory_seq_rm(mem, 0, sp.first, sp.second);
            turn_hi = (int)pos;
        }

        // commit the turn: one segment = user line + answer (crash-safe, every turn)
        auto buf = spill_range(mem, turn_lo, turn_hi);
        int new_id = 0; for (auto &rr : rows) new_id = std::max(new_id, rr.id + 1); // ids sparse after prune
        CatRow r; r.id = new_id; r.lo = turn_lo; r.hi = turn_hi; r.w = 0.f;
        r.text = (used_fallback ? "\nUser: " + q + "\nAssistant: " + out : "\nUser: " + q + "\nAssistant:" + out);
        write_file(dir+"/segments/"+std::to_string(r.id)+".kv", buf.data(), buf.size());
        auto e = embed(r.text, false);
        write_file(dir+"/embs.f32", e.data(), EMB_DIM*4, "ab");
        embraw.append((const char*)e.data(), EMB_DIM*4);
        { FILE *f = fopen((dir+"/catalog.tsv").c_str(), "a"); if (!f) fail("catalog write");
          fprintf(f, "%d\t%d\t%d\t%.5f\t%s\n", r.id, r.lo, r.hi, r.w, tsv_escape(r.text).c_str()); fclose(f); }
        rows.push_back(r);
        m.total_tokens = turn_hi; m.n_segments++; m.emb_dim = EMB_DIM;
        save_manifest(dir, m);

        // live-window budget: evict oldest live segments once the window outgrows ~NCTX/2
        while (m.total_tokens - live_lo > NCTX/2 && !rows.empty()) {
            for (auto &rr : rows) if (rr.lo == live_lo) { llama_memory_seq_rm(mem, 0, rr.lo, rr.hi); live_lo = rr.hi; goto evicted; }
            break; evicted:;
        }
    }
    free(line);
    save_catalog(dir, rows); // persist w updates
    fprintf(stderr, "[chat] saved: %ld tokens, %ld segments\n", m.total_tokens, m.n_segments);
    return 0;
}

// prune (M3): sink cold segments. Their KV is deleted and the catalog row moves to archive.tsv
// — the TEXT is never lost (re-ingest from archive if ever needed). Surviving rows keep their
// ids and positions; run `kvmem defrag` afterwards to actually compact the token space.
static int cmd_prune(const std::string &dir, const std::string &spec) {
    Manifest m; if (!load_manifest(dir, m)) fail("no store at " + dir);
    apply_emb(m);
    auto rows = load_catalog(dir);
    if ((long)rows.size() != m.n_segments) fail("catalog/manifest mismatch");
    std::string embraw = read_file(dir+"/embs.f32");
    if (embraw.size() != rows.size()*EMB_DIM*4) fail("embs.f32 size mismatch");

    std::vector<char> drop(rows.size(), 0);
    if (spec.rfind("keep=", 0) == 0) {
        int N = atoi(spec.c_str()+5); if (N < 1) fail("keep=N needs N >= 1");
        std::vector<int> idx(rows.size()); for (size_t i=0;i<rows.size();i++) idx[i]=(int)i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a,int b){ return rows[a].w > rows[b].w; });
        for (size_t k = N; k < idx.size(); k++) drop[idx[k]] = 1;
    } else if (spec.rfind("below=", 0) == 0) {
        float W = (float)atof(spec.c_str()+6);
        size_t kept = 0;
        for (size_t i=0;i<rows.size();i++) { if (rows[i].w < W) drop[i]=1; else kept++; }
        if (!kept) fail("below=" + std::to_string(W) + " would drop everything");
    } else fail("prune spec: keep=N (hottest) or below=W (drop w < W)");

    long n_drop = 0; for (char d : drop) n_drop += d;
    if (!n_drop) { fprintf(stderr, "[prune] nothing to drop\n"); return 0; }

    FILE *a = fopen((dir+"/archive.tsv").c_str(), "a"); if (!a) fail("cannot write archive.tsv");
    std::vector<CatRow> keep; std::string kembs; kembs.reserve((rows.size()-n_drop)*EMB_DIM*4);
    for (size_t i = 0; i < rows.size(); i++) {
        if (drop[i]) {
            fprintf(a, "%d\t%d\t%d\t%.5f\t%s\n", rows[i].id, rows[i].lo, rows[i].hi, rows[i].w, tsv_escape(rows[i].text).c_str());
            unlink((dir+"/segments/"+std::to_string(rows[i].id)+".kv").c_str());
        } else {
            keep.push_back(rows[i]);
            kembs.append(embraw, i*EMB_DIM*4, EMB_DIM*4);
        }
    }
    fclose(a);
    write_file(dir+"/embs.f32", kembs.data(), kembs.size());
    save_catalog(dir, keep);
    m.n_segments = (long)keep.size();
    save_manifest(dir, m);
    fprintf(stderr, "[prune] dropped %ld of %zu segments (text kept in archive.tsv); run `kvmem defrag %s` to compact\n",
            n_drop, rows.size(), dir.c_str());
    return 0;
}

// defrag (M2): rebuild the store from the STORED TEXT — positions restart at 0 (cures the
// monotonic-position ceiling) and the rebuild runs on the CURRENT MODEL (cures model-lock:
// set MODEL=new.gguf to migrate). nomic embeddings are model-independent -> embs.f32 is
// copied as-is, no ollama needed. Atomic: build in <dir>.new, swap via rename, old -> .bak.
static int cmd_defrag(const std::string &dir) {
    Manifest mold; if (!load_manifest(dir, mold)) fail("no store at " + dir);
    auto old_rows = load_catalog(dir);
    if ((long)old_rows.size() != mold.n_segments) fail("catalog/manifest mismatch");
    const std::string model = env_s("MODEL", "models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf");
    const std::string kvt = env_s("KVT", mold.kv_type.c_str());
    const int NCTX=env_i("NCTX",4096), CHUNK=env_i("CHUNK",512), WIN=env_i("WIN",256);
    struct stat st; if (stat(model.c_str(), &st)) fail("model not found: " + model);

    std::string ndir = dir + ".new", bdir = dir + ".bak";
    if (!stat(ndir.c_str(), &st)) fail(ndir + " exists — remove it first");
    struct stat stb; if (!stat(bdir.c_str(), &stb)) fail(bdir + " exists (previous defrag backup) — remove it first");
    mkdir(ndir.c_str(), 0755); mkdir((ndir+"/segments").c_str(), 0755);

    load_model(model, NCTX, CHUNK, kvt);
    // same segments, same order, compact positions from 0
    std::vector<llama_token> ids; std::vector<CatRow> rows;
    for (size_t s = 0; s < old_rows.size(); s++) {
        auto tk = tokenize(old_rows[s].text, s == 0);
        CatRow r; r.id = (int)s; r.lo = (int)ids.size();
        ids.insert(ids.end(), tk.begin(), tk.end());
        r.hi = (int)ids.size(); r.w = old_rows[s].w; r.text = old_rows[s].text; // w carries over
        rows.push_back(r);
    }
    fprintf(stderr, "[defrag] %zu segments: %ld tokens -> %zu (model %s)\n",
            rows.size(), mold.total_tokens, ids.size(), model.c_str());
    double secs = prefill_spill(ndir, rows, ids, 0, CHUNK, WIN);
    llama_free(g_ctx); llama_model_free(g_model); g_ctx = nullptr;

    // embeddings depend only on text (unchanged) -> straight copy
    std::string embraw = read_file(dir + "/embs.f32");
    write_file(ndir + "/embs.f32", embraw.data(), embraw.size());
    save_catalog(ndir, rows);
    struct stat stm; stat(model.c_str(), &stm);
    Manifest m; m.model_file = model.substr(model.find_last_of('/')+1); m.model_bytes = stm.st_size;
    m.kv_type = kvt; m.build = env_s("KVMEM_BUILD", "b9297");
    m.total_tokens = (long)ids.size(); m.n_segments = (long)rows.size();
    m.emb_model = mold.emb_model; m.emb_dim = mold.emb_dim;   // embs.f32 is copied verbatim
    save_manifest(ndir, m);

    if (rename(dir.c_str(), bdir.c_str())) fail("swap failed: cannot move " + dir + " aside");
    if (rename(ndir.c_str(), dir.c_str())) { rename(bdir.c_str(), dir.c_str()); fail("swap failed: rollback done"); }
    fprintf(stderr, "[defrag] done in %.1fs: %ld -> %ld tokens, old store kept at %s\n",
            secs, mold.total_tokens, m.total_tokens, bdir.c_str());
    return 0;
}

// serve (M3): keep the model RESIDENT and answer over plain HTTP (localhost only, no deps).
// One-shot `ask` pays ~seconds of model load per question; serve pays it once.
//   curl -s localhost:8345/stats
//   curl -s -d "what's the gate code?" localhost:8345/ask     (answer in body;
//        X-Low-Confidence: 1 header when the probe score is below THRESH)
//   curl -s -X POST localhost:8345/shutdown                   (or SIGTERM — both save w)
static volatile sig_atomic_t g_srv_stop = 0;
static void srv_on_term(int) { g_srv_stop = 1; }

static int cmd_serve(const std::string &dir) {
    Manifest m; std::vector<CatRow> rows; std::string embraw, model;
    open_store(dir, m, rows, embraw, model);
    g_emb_cpu = env_i("EMB_CPU", 1) != 0;     // model stays resident -> nomic embeds on CPU
    load_model(model, env_i("NCTX",4096), env_i("CHUNK",512), m.kv_type);
    llama_memory_t mem = llama_get_memory(g_ctx);

    int port = env_i("KVPORT", 8345);
    int s = socket(AF_INET, SOCK_STREAM, 0); if (s < 0) fail("socket failed");
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);                 // localhost ONLY
    if (bind(s, (sockaddr*)&a, sizeof a)) fail("bind 127.0.0.1:" + std::to_string(port) + " failed (port busy? set KVPORT)");
    if (listen(s, 8)) fail("listen failed");
    struct sigaction sa{}; sa.sa_handler = srv_on_term; sigaction(SIGTERM, &sa, nullptr); sigaction(SIGINT, &sa, nullptr);
    fprintf(stderr, "[serve] %s ready on 127.0.0.1:%d — POST /ask, GET /stats, POST /shutdown\n", dir.c_str(), port);

    while (!g_srv_stop) {
        int c = accept(s, nullptr, nullptr);
        if (c < 0) { if (errno == EINTR) continue; break; }
        std::string req; char buf[8192]; ssize_t n;
        size_t hdr_end = std::string::npos; long clen = 0;
        while ((n = recv(c, buf, sizeof buf, 0)) > 0) {
            req.append(buf, n);
            if (hdr_end == std::string::npos) {
                hdr_end = req.find("\r\n\r\n");
                if (hdr_end != std::string::npos) {
                    size_t p = req.find("Content-Length:"); // also matches lowercase via retry below
                    if (p == std::string::npos) p = req.find("content-length:");
                    if (p != std::string::npos && p < hdr_end) clen = atol(req.c_str() + p + 15);
                }
            }
            if (hdr_end != std::string::npos && req.size() >= hdr_end + 4 + (size_t)clen) break;
        }
        if (hdr_end == std::string::npos) { close(c); continue; }
        std::string method = req.substr(0, req.find(' '));
        size_t ps = req.find(' ') + 1;
        std::string path = req.substr(ps, req.find(' ', ps) - ps);
        std::string body = req.substr(hdr_end + 4, clen);

        int code = 200; std::string out, xtra;
        if (path == "/stats") {
            char hb[512];
            snprintf(hb, sizeof hb, "model=%s kv=%s tokens=%ld segments=%ld\n", m.model_file.c_str(), m.kv_type.c_str(), m.total_tokens, m.n_segments);
            out = hb;
            auto byw = rows; std::sort(byw.begin(), byw.end(), [](const CatRow&x, const CatRow&y){ return x.w > y.w; });
            for (int i = 0; i < 5 && i < (int)byw.size(); i++)
                out += "seg" + std::to_string(byw[i].id) + " w=" + std::to_string(byw[i].w).substr(0,5) + "  " + byw[i].text.substr(0, 70) + "...\n";
        } else if (path == "/ask" && method == "POST" && !body.empty()) {
            llama_memory_clear(mem, true);                 // each ask starts from a clean cache
            bool low = false;
            out = ask_core(dir, m, rows, embraw, body, low) + "\n";
            xtra = std::string("X-Low-Confidence: ") + (low ? "1" : "0") + "\r\n";
        } else if (path == "/shutdown" && method == "POST") {
            out = "bye\n"; g_srv_stop = 1;
        } else { code = 404; out = "kvmem serve: POST /ask (body=question), GET /stats, POST /shutdown\n"; }

        char hdr[512];
        int hn = snprintf(hdr, sizeof hdr, "HTTP/1.1 %d %s\r\nContent-Type: text/plain; charset=utf-8\r\n%sContent-Length: %zu\r\nConnection: close\r\n\r\n",
                          code, code == 200 ? "OK" : "Not Found", xtra.c_str(), out.size());
        send(c, hdr, hn, MSG_NOSIGNAL); send(c, out.data(), out.size(), MSG_NOSIGNAL);
        close(c);
    }
    close(s);
    save_catalog(dir, rows);                               // w already saved per ask; belt for SIGTERM mid-flight
    fprintf(stderr, "[serve] stopped, catalog saved\n");
    return 0;
}

// reembed (M6): rewrite embs.f32 with a different ollama embedding model (EMB=bge-m3 etc.).
// Text and KV are untouched — embeddings are catalog-side. The 7B is never loaded, so the
// embedder may use the GPU (EMB_CPU=0 default here, unlike chat/serve where the 7B is resident).
static int cmd_reembed(const std::string &dir) {
    Manifest m; if (!load_manifest(dir, m)) fail("no store at " + dir);
    auto rows = load_catalog(dir);
    if ((long)rows.size() != m.n_segments) fail("catalog/manifest mismatch");
    std::string target = env_s("EMB", "");
    if (target.empty()) fail("set EMB=<ollama embedding model>, e.g. EMB=bge-m3");
    fprintf(stderr, "[reembed] %zu segments: %s -> %s\n", rows.size(), m.emb_model.c_str(), target.c_str());
    EMB_MODEL = target; EMB_DIM = 0;                   // learn the new dim from the first reply
    g_emb_cpu = env_i("EMB_CPU", 0) != 0;
    std::string embs; embs.reserve(rows.size() * 1024 * 4);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < rows.size(); i++) {
        auto e = embed(rows[i].text, false);
        embs.append((const char*)e.data(), EMB_DIM*4);
        if ((i+1) % 1000 == 0) {
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
            fprintf(stderr, "[reembed] %zu/%zu (%.0f seg/s)\n", i+1, rows.size(), (i+1)/dt);
        }
    }
    // verified tmp+rename — same ENOSPC discipline as save_catalog (2026-06-06 lesson)
    std::string tmp = dir + "/embs.f32.tmp";
    FILE *f = fopen(tmp.c_str(), "wb"); if (!f) fail("cannot write " + tmp);
    bool bad = embs.size() && fwrite(embs.data(), 1, embs.size(), f) != embs.size();
    if (fflush(f) != 0 || ferror(f)) bad = true;
    if (fclose(f) != 0) bad = true;
    if (bad) { remove(tmp.c_str()); fail("embs save failed (disk full?) — old embs.f32 kept"); }
    if (rename(tmp.c_str(), (dir + "/embs.f32").c_str()) != 0) fail("embs rename failed — old embs.f32 kept");
    m.emb_model = EMB_MODEL; m.emb_dim = EMB_DIM;
    save_manifest(dir, m);
    fprintf(stderr, "[reembed] done: %s dim=%d, %zu segments\n", EMB_MODEL.c_str(), EMB_DIM, rows.size());
    return 0;
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
                        "  kvmem probe  <store_dir> \"question\"   # GPU-free gap sensor: scores only, no generation\n"
                        "  kvmem chat   <store_dir>            # interactive; later sessions remember\n"
                        "  kvmem serve  <store_dir>            # resident model + HTTP /ask /stats (KVPORT=8345)\n"
                        "  kvmem prune  <store_dir> keep=N|below=W  # sink cold segments (text -> archive.tsv)\n"
                        "  kvmem defrag <store_dir>            # rebuild from stored text (MODEL=... migrates)\n"
                        "  kvmem reembed <store_dir>           # rewrite embs.f32 with EMB=<model> (bge-m3 etc.)\n"
                        "  kvmem stats  <store_dir>\n"
                        "env: MODEL=<gguf> KVT=q8_0 NCTX=4096 SEL=3 GAP=0.04 THRESH=0.65 GEN=96 COPYB=2.0 KVSHIFT=1\n"
                        "     EXTRACT=1 (verbatim stored text)  GROUND=1 (answer in own words from probed text)\n"
                        "     EMB=<ollama embedding model> (new stores / reembed)\n");
        return 1;
    }
    llama_log_set([](ggml_log_level l, const char *t, void*){ if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    std::string cmd = argv[1], dir = argv[2];
    if (cmd == "ingest" && argc > 3) return cmd_ingest(dir, argv[3]);
    if (cmd == "ask"    && argc > 3) return cmd_ask(dir, argv[3]);
    if (cmd == "probe"  && argc > 3) return cmd_probe(dir, argv[3]);
    if (cmd == "chat")               return cmd_chat(dir);
    if (cmd == "serve")              return cmd_serve(dir);
    if (cmd == "prune"  && argc > 3) return cmd_prune(dir, argv[3]);
    if (cmd == "defrag")             return cmd_defrag(dir);
    if (cmd == "reembed")            return cmd_reembed(dir);
    if (cmd == "stats")              return cmd_stats(dir);
    fprintf(stderr, "kvmem: bad arguments\n"); return 1;
}
// compile:
// g++ -O2 -o kvmem kvmem.cpp -I <llama.cpp>/include -I <llama.cpp>/ggml/include \
//     -L <llama.cpp>/build/bin -lllama -Wl,-rpath,<llama.cpp>/build/bin
