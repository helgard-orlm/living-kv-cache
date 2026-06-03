"""
THE live test: load helgard's ENTIRE wiki into Qwen's KV cache, persist it, answer real questions
from the reloaded cache without re-reading the wiki.

  mode=build : read wiki_pages.jsonl -> token ids + paragraph segments (with page headers) ->
               windowed prefill -> save cache (cold store + catalog) + meta (seg->page map).
  mode=ask   : load cache (NO prefill) -> answer questions from questions.json [{"q","ans"}] ->
               rank/prob of the answer's first token + which page the probe picked.

  MODE=build JSONL=/tmp/wiki_pages.jsonl CACHE=/data/wiki_kv_8bit.pt BITS=8 python wiki_kv_demo.py
  MODE=ask   CACHE=/data/wiki_kv_8bit.pt QUESTIONS=/tmp/wiki_q.json python wiki_kv_demo.py
"""
import os, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
sys.path.insert(0, os.path.expanduser("~/research/living-kv-cache"))
from living_kv import LivingKVRunner

DEV = "cuda"
MID = os.environ.get("MODEL_DIR", "Qwen/Qwen2.5-7B-Instruct-1M")
MODE = os.environ.get("MODE", "build")
JSONL = os.environ.get("JSONL", "/tmp/wiki_pages.jsonl")
CACHE = os.environ.get("CACHE", "/data/wiki_kv_8bit.pt")
QUESTIONS = os.environ.get("QUESTIONS", "/tmp/wiki_q.json")
BITS = int(os.environ.get("BITS", "8"))
WIN = int(os.environ.get("WIN", "4096"))
CHUNK = int(os.environ.get("CHUNK", "512"))
MAXN = int(os.environ.get("MAXN", "780000"))   # stay under proven 800k / RAM

def load_model():
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, attn_implementation="sdpa",
        device_map=DEV, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)).eval()
    return model, tok

def build():
    model, tok = load_model()
    ids, segs, seg_pages = [], [], []
    def add(text, page, first=False):
        e = tok.encode(text, add_special_tokens=first)
        if len(ids) + len(e) > MAXN: return False
        segs.append((len(ids), len(ids) + len(e))); seg_pages.append(page); ids.extend(e)
        return True
    add(" Wiki dump.\n", "_header", first=True)
    n_pages = 0; truncated = False
    with open(JSONL) as f:
        for ln in f:
            d = json.loads(ln)
            hdr = f"\n\n# PAGE: {d['path']} — {d['title']}\n"
            if not add(hdr, d["path"]): truncated = True; break
            for para in [p.strip() for p in d["content"].split("\n\n") if p.strip()]:
                if not add(" " + para + "\n", d["path"]): truncated = True; break
            if truncated: break
            n_pages += 1
    print(f"corpus: {len(ids):,} tokens | {len(segs):,} segments | {n_pages} pages{' (TRUNCATED at MAXN)' if truncated else ''}", flush=True)
    runner = LivingKVRunner(model, tok, win=WIN, bits=BITS, device=DEV)
    t0 = time.time()
    runner.prefill(ids, segs, chunk=CHUNK)
    print(f"prefill took {time.time()-t0:.0f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    t1 = time.time()
    runner.save(CACHE)
    json.dump({"seg_pages": seg_pages, "n_tokens": len(ids), "n_pages": n_pages},
              open(CACHE + ".meta.json", "w"))
    print(f"saved {os.path.getsize(CACHE)/1e9:.1f}GB cache in {time.time()-t1:.0f}s -> {CACHE}", flush=True)

def ask():
    model, tok = load_model()
    t0 = time.time()
    runner = LivingKVRunner.load(CACHE, model, tok, device=DEV)
    meta = json.load(open(CACHE + ".meta.json"))
    rmpc = int(os.environ.get("RMPC", "-1"))
    if rmpc >= 0 and hasattr(runner, "_segvec"):
        runner.rebuild_catalog(rmpc)
        print(f"catalog rebuilt with remove_pc={rmpc}", flush=True)
    print(f"cache loaded in {time.time()-t0:.0f}s: {meta['n_tokens']:,} tokens / {meta['n_pages']} pages (NO prefill this session)", flush=True)
    bud=int(os.environ.get("BUDGET","0"));
    if bud>0: runner.budget=bud; print(f"budget={bud}",flush=True)
    qs = json.load(open(QUESTIONS))
    hits = 0
    diag = os.environ.get("DIAG", "0") == "1"
    for item in qs:
        rank, prob, seg = runner.answer(item["q"], item["ans"])
        page = meta["seg_pages"][seg] if seg is not None and seg < len(meta["seg_pages"]) else "?"
        ok = rank <= 1
        hits += int(ok)
        print(f"  [{'OK ' if ok else 'MISS'}] rank {rank:<6} p {prob:.3f} | probe->{page[:48]:<48} | {item['q'][:60]!r} -> {item['ans']!r}", flush=True)
    print(f"\n{hits}/{len(qs)} answered at rank<=1 from the RELOADED wiki cache", flush=True)
    if diag:
        import torch as _t
        print("\n=== probe top-5 diagnostic (per question) ===", flush=True)
        for item in qs:
            qids2 = tok.encode(item["q"], add_special_tokens=False)
            runner._mode = "probe"
            with _t.no_grad():
                oo = model(input_ids=_t.tensor([qids2], device=DEV),
                           position_ids=_t.arange(len(qids2), device=DEV)[None],
                           use_cache=False, output_hidden_states=True)
            qe = runner._deanis(oo.hidden_states[runner.lmid][0, -1:].float().cpu())[0]
            ss = runner._seg_emb @ qe
            top = _t.topk(ss, 5)
            cand = ", ".join(f"{meta['seg_pages'][i][:24]}({v:.2f})" for v, i in zip(top.values.tolist(), top.indices.tolist()))
            print(f"  Q={item['q'][:45]!r}: {cand}", flush=True)

if MODE == "build": build()
else: ask()
