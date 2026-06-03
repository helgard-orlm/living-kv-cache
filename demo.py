"""
Needle-in-a-haystack demo for living-kv-cache.

Plants a fact near the START of a long synthetic context, fills the rest with distractors, then asks
for the fact. Measures whether the spill (small hot window in VRAM + cold KV in RAM + segment probe)
still recalls it. Reproduces the scaling table in the README.

Defaults to Qwen2.5-7B-Instruct-1M in 4-bit on an 8GB GPU. Override with env vars.

    MODEL_DIR=Qwen/Qwen2.5-7B-Instruct-1M N=200000 BITS=8 python demo.py

Honest scope: single synthetic needle, windowed prefill (late tokens see only a WIN-token window),
4-bit cold store is NOT lossless (use 8 or 16). See README.
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from living_kv import LivingKVRunner

DEV = "cuda"
MID = os.environ.get("MODEL_DIR", "Qwen/Qwen2.5-7B-Instruct-1M")
N = int(os.environ.get("N", "200000"))
BITS = int(os.environ.get("BITS", "16"))      # 16 fp16 | 8 lossless 2x | 4 lossy
WIN = int(os.environ.get("WIN", "4096"))
BUDGET = int(os.environ.get("BUDGET", "256"))
CHUNK = int(os.environ.get("CHUNK", "512"))
LMID = int(os.environ.get("LMID", "18"))
LOAD_4BIT = os.environ.get("LOAD_4BIT", "1") == "1"

FACT = " Important: my computer password is bluefalcon."
Q = " Now, what is my computer password? My computer password is"
ANS = " blue"
DIST = [
    " The weather was grey and damp with drizzle and a cold breeze at night.",
    " Boats rocked against the wooden pier under a pale quiet sky and gulls called.",
    " People walked past the cafe near the corner talking quietly about nothing.",
    " On Monday it rained, by Tuesday the roads were dry, Wednesday turned warm.",
    " The shelves were full of dusty books that nobody had opened in many years.",
    " My old password used to be redhawk before I changed it a long time ago.",
]

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    kw = dict(dtype=torch.float16, attn_implementation="sdpa", device_map=DEV)
    if LOAD_4BIT:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MID, **kw).eval()
    print(f"model {MID} | weights VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB | N {N} WIN {WIN} budget {BUDGET} bits {BITS}", flush=True)

    # build context: " Notes." + FACT (early needle) + distractor filler to length N
    segs, ids = [], []
    def add(t, first=False):
        e = tok.encode(t, add_special_tokens=first)
        segs.append((len(ids), len(ids) + len(e))); ids.extend(e)
    add(" Notes.", True); add(FACT)
    di = 0
    while len(ids) < N:
        add(DIST[di % len(DIST)]); di += 1
    ids = ids[:N]; segs = [(lo, min(hi, N)) for lo, hi in segs if lo < N]
    print(f"tokens {len(ids)} segments {len(segs)} needle span {segs[1]}", flush=True)

    runner = LivingKVRunner(model, tok, win=WIN, budget=BUDGET, lmid=LMID, bits=BITS, device=DEV)
    runner.prefill(ids, segs, chunk=CHUNK)
    print(f"prefill done. cold store ~{runner.cold_store_gb():.1f}GB RAM | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    rank, prob, seg = runner.answer(Q, ANS)
    ok = (seg == 1)
    print(f"\nRESULT  N={len(ids)}  rank={rank}  prob={prob:.4f}  probe_picked_seg={seg} (needle=seg1, {'HIT' if ok else 'miss'})", flush=True)

if __name__ == "__main__":
    main()
