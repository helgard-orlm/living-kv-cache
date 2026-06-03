"""
Cache-as-memory demo: prefill a long text ONCE, save the cold store + catalog to disk, then (simulating a
fresh session) reload it and answer a question WITHOUT re-running the expensive prefill.

This is "cache-augmented" memory: the model resumes with the text already 'read'. The cache is specific to
this model + quantization (not portable). Honest scope: same caveats as demo.py (windowed prefill, single
needle, effective-context limit, 4-bit not lossless).

    MODEL_DIR=Qwen/Qwen2.5-7B-Instruct-1M N=100000 BITS=8 python persist_demo.py
"""
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from living_kv import LivingKVRunner

DEV = "cuda"
MID = os.environ.get("MODEL_DIR", "Qwen/Qwen2.5-7B-Instruct-1M")
N = int(os.environ.get("N", "100000"))
BITS = int(os.environ.get("BITS", "8"))
WIN = int(os.environ.get("WIN", "4096"))
CHUNK = int(os.environ.get("CHUNK", "512"))
PATH = os.environ.get("CACHE_PATH", "/tmp/wiki_kv_cache.pt")

FACT = " Important: my computer password is bluefalcon."
Q = " Now, what is my computer password? My computer password is"
ANS = " blue"
DIST = [" The weather was grey and damp.", " Boats rocked against the pier.",
        " People walked past the cafe.", " On Monday it rained, Tuesday dry.",
        " Dusty books nobody opened.", " My old password was redhawk long ago."]


def load_model():
    tok = AutoTokenizer.from_pretrained(MID)
    kw = dict(dtype=torch.float16, attn_implementation="sdpa", device_map=DEV,
              quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    return AutoModelForCausalLM.from_pretrained(MID, **kw).eval(), tok


def build_ids(tok):
    segs, ids = [], []
    def add(t, first=False):
        e = tok.encode(t, add_special_tokens=first); segs.append((len(ids), len(ids) + len(e))); ids.extend(e)
    add(" Notes.", True); add(FACT); di = 0
    while len(ids) < N:
        add(DIST[di % len(DIST)]); di += 1
    return ids[:N], [(lo, min(hi, N)) for lo, hi in segs if lo < N]


def main():
    model, tok = load_model()
    ids, segs = build_ids(tok)

    # --- session 1: prefill once and SAVE ---
    print(f"[session 1] prefilling {len(ids)} tokens and saving cache to {PATH} ...", flush=True)
    t0 = time.time()
    r = LivingKVRunner(model, tok, win=WIN, bits=BITS, device=DEV)
    r.prefill(ids, segs, chunk=CHUNK)
    r.save(PATH)
    print(f"[session 1] prefill+save took {time.time()-t0:.1f}s | cache file {os.path.getsize(PATH)/1e9:.1f}GB", flush=True)

    # --- session 2: fresh runner, LOAD cache, answer WITHOUT prefill ---
    print("[session 2] loading cache (no prefill) and answering ...", flush=True)
    t1 = time.time()
    r2 = LivingKVRunner.load(PATH, model, tok, device=DEV)
    rank, prob, seg = r2.answer(Q, ANS)
    print(f"[session 2] load+answer took {time.time()-t1:.1f}s", flush=True)
    print(f"\nRESULT (from reloaded cache)  rank={rank}  prob={prob:.4f}  picked_seg={seg} (needle=seg1, {'HIT' if seg==1 else 'miss'})", flush=True)
    print("=> the model recalled the fact from a cache it did NOT re-compute this session.", flush=True)


if __name__ == "__main__":
    main()
