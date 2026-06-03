"""
Living KV-cache spill — core runner.

Idea: keep only a small HOT window of KV in GPU VRAM; stream the rest ("cold" KV) to cheap CPU RAM.
On a query, a cheap in-VRAM catalog (segment self-embeddings) finds the relevant cold segment and pulls
only its KV back. VRAM stays small and constant regardless of context length; the cold store lives in RAM
(optionally 8-bit compressed). Positions are correct by construction: keys arrive already RoPE-rotated at
their absolute position, and we pass explicit position_ids per chunk.

Works on any HuggingFace full-attention transformer with a normal KV cache (Llama/Qwen2/Mistral/GLM/...).
Does NOT apply to Mamba/SSM/linear-attention hybrids (no growing KV to spill).

This is a research PoC, not a production library. See README for honest caveats.
"""
import torch
import torch.nn.functional as F
from transformers import AttentionInterface

NEG = None  # set per-dtype in LivingKVRunner


def _hadamard(n, device):
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def _repeat_kv(hs, n):
    b, k, s, d = hs.shape
    if n == 1:
        return hs
    return hs[:, :, None, :, :].expand(b, k, n, s, d).reshape(b, k * n, s, d)


class LivingKVRunner:
    """
    Usage:
        runner = LivingKVRunner(model, tokenizer, win=4096, budget=256, lmid=18, bits=8)
        runner.prefill(token_ids)                 # windowed streaming prefill -> cold store + catalog
        rank, prob, seg = runner.answer(question, answer_token)

    bits: cold-store quantization. 16 = fp16 (lossless, ~2 bytes/val). 8 = WHT + 8-bit (lossless recall,
          ~2x RAM saving). 4 = WHT + 4-bit (NOT lossless on long context — see README).
    """

    def __init__(self, model, tokenizer, win=4096, budget=256, recency=32,
                 lmid=18, remove_pc=2, bits=16, device="cuda"):
        self.model = model.eval()
        self.tok = tokenizer
        self.dev = device
        self.dt = next(model.parameters()).dtype if model.dtype is None else model.dtype
        cfg = model.config
        self.NL = cfg.num_hidden_layers
        self.G = cfg.num_attention_heads // cfg.num_key_value_heads
        self.NKV = cfg.num_key_value_heads
        self.HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.hidden = cfg.hidden_size
        self.win = win
        self.budget = budget
        self.recency = recency
        self.lmid = lmid
        self.remove_pc = remove_pc
        self.bits = bits
        self.lv = 2 ** (bits - 1) - 1 if bits < 16 else None
        self.neg = torch.finfo(self.dt).min
        self.H = _hadamard(self.HD, device) if bits < 16 else None
        # state
        self._mode = "prefill"
        self._winK = [None] * self.NL
        self._winV = [None] * self.NL
        self._coldK = [None] * self.NL  # preallocated (1,NKV,N,HD) fp16  OR int8 codes if quantized
        self._coldV = [None] * self.NL
        self._coldKs = [None] * self.NL  # scales (quant only)
        self._coldVs = [None] * self.NL
        self._wpos = 0  # write cursor into cold store
        self._ansK = [None] * self.NL
        self._ansV = [None] * self.NL
        self._impl_name = f"living_spill_{id(self)}"
        AttentionInterface.register(self._impl_name, self._attn)
        model.config._attn_implementation = self._impl_name

    # ---- quantization (TurboQuant-style: WHT rotate + scalar quant) ----
    def _enc(self, x):  # x (1,NKV,n,HD) -> codes int8 cpu, scale cpu
        xr = x.float() @ self.H
        s = xr.abs().amax(-1, keepdim=True).clamp_min(1e-8)
        code = torch.round(xr / s * self.lv).clamp(-self.lv, self.lv).to(torch.int8)
        return code.cpu(), s.to(self.dt).cpu()

    def _dec(self, code, s):  # -> (1,NKV,n,HD) on dev
        xr = (code.to(self.dev).float() / self.lv) * s.to(self.dev).float()
        return (xr @ self.H).to(self.dt)

    def _causal(self, n, w):
        qp = torch.arange(w - n, w, device=self.dev)[:, None]
        kp = torch.arange(w, device=self.dev)[None, :]
        return torch.where(kp <= qp, 0.0, self.neg).to(self.dt)[None, None]

    # ---- the custom attention dispatched by transformers ----
    def _attn(self, module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if scaling is None:
            scaling = query.size(-1) ** -0.5
        l = module.layer_idx
        n = key.shape[-2]
        if self._mode == "probe":  # self-attention over current tokens only (for question hidden)
            Kg = _repeat_kv(key, self.G); Vg = _repeat_kv(value, self.G)
            return F.scaled_dot_product_attention(query, Kg, Vg, attn_mask=self._causal(n, n), scale=scaling).transpose(1, 2), None
        if self._mode == "prefill":
            # write to preallocated cold store
            if self.bits < 16:
                kc, ks = self._enc(key); vc, vs = self._enc(value)
                self._coldK[l][:, :, self._wpos0:self._wpos0 + n, :] = kc
                self._coldV[l][:, :, self._wpos0:self._wpos0 + n, :] = vc
                self._coldKs[l][:, :, self._wpos0:self._wpos0 + n, :] = ks
                self._coldVs[l][:, :, self._wpos0:self._wpos0 + n, :] = vs
            else:
                self._coldK[l][:, :, self._wpos0:self._wpos0 + n, :] = key.detach().cpu()
                self._coldV[l][:, :, self._wpos0:self._wpos0 + n, :] = value.detach().cpu()
            # GPU sliding window
            wk = key if self._winK[l] is None else torch.cat([self._winK[l], key], 2)
            wv = value if self._winV[l] is None else torch.cat([self._winV[l], value], 2)
            if wk.shape[2] > self.win:
                wk = wk[:, :, -self.win:, :].contiguous(); wv = wv[:, :, -self.win:, :].contiguous()
            self._winK[l] = wk; self._winV[l] = wv
            w = wk.shape[2]
            Kg = _repeat_kv(wk, self.G); Vg = _repeat_kv(wv, self.G)
            return F.scaled_dot_product_attention(query, Kg, Vg, attn_mask=self._causal(n, w), scale=scaling).transpose(1, 2), None
        # answer: attend to gathered budget + the question's own tokens
        Kc = self._ansK[l]; Vc = self._ansV[l]; kb = Kc.shape[2]
        Kg = _repeat_kv(torch.cat([Kc, key], 2), self.G); Vg = _repeat_kv(torch.cat([Vc, value], 2), self.G)
        am = torch.zeros(1, 1, n, kb + n, device=self.dev, dtype=self.dt)
        qr = torch.arange(n, device=self.dev)
        am[..., kb:] = torch.where(qr[None, :] <= qr[:, None], 0.0, self.neg).to(self.dt)[None, None]
        return F.scaled_dot_product_attention(query, Kg, Vg, attn_mask=am, scale=scaling).transpose(1, 2), None

    # ---- public API ----
    def prefill(self, ids, segments, chunk=512):
        """ids: list[int]. segments: list[(lo,hi)] token spans for the catalog (e.g. sentences)."""
        self.N = len(ids)
        self.segments = segments
        # preallocate cold store
        store_dt = torch.int8 if self.bits < 16 else self.dt
        for l in range(self.NL):
            self._coldK[l] = torch.empty(1, self.NKV, self.N, self.HD, dtype=store_dt)
            self._coldV[l] = torch.empty(1, self.NKV, self.N, self.HD, dtype=store_dt)
            if self.bits < 16:
                self._coldKs[l] = torch.empty(1, self.NKV, self.N, 1, dtype=self.dt)
                self._coldVs[l] = torch.empty(1, self.NKV, self.N, 1, dtype=self.dt)
            self._winK[l] = None; self._winV[l] = None
        segsum = torch.zeros(len(segments), self.hidden); segcnt = torch.zeros(len(segments))
        self._mode = "prefill"
        with torch.no_grad():
            for i in range(0, self.N, chunk):
                ch = ids[i:i + chunk]; n = len(ch)
                self._wpos0 = i
                o = self.model(input_ids=torch.tensor([ch], device=self.dev),
                               position_ids=torch.arange(i, i + n, device=self.dev)[None],
                               use_cache=False, output_hidden_states=True)
                h = o.hidden_states[self.lmid][0].float().cpu()
                for s, (lo, hi) in enumerate(segments):
                    a = max(lo, i); b = min(hi, i + n)
                    if a < b:
                        segsum[s] += h[a - i:b - i].sum(0); segcnt[s] += (b - a)
                del o, h
        self._segvec = (segsum / segcnt.clamp_min(1).unsqueeze(1)).float()  # raw CPU catalog (kept for re-tuning)
        self.rebuild_catalog(self.remove_pc)

    def rebuild_catalog(self, remove_pc):
        """Re-derive the probe catalog from the raw segment vectors with a different top-PC removal.
        Cheap (CPU SVD) — lets you tune the probe on a saved cache without re-prefilling."""
        self.remove_pc = remove_pc
        mean = self._segvec.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(self._segvec - mean, full_matrices=False)
        Vpc = Vh[:remove_pc].T if remove_pc > 0 else None
        self._mean, self._Vpc = mean, Vpc
        self._seg_emb = self._deanis(self._segvec)

    def _deanis(self, x):
        x = x - self._mean.squeeze(0)
        if self._Vpc is not None:
            x = x - (x @ self._Vpc) @ self._Vpc.T
        return F.normalize(x, dim=-1)

    def _probe(self, question):
        qids = self.tok.encode(question, add_special_tokens=False)
        self._mode = "probe"
        with torch.no_grad():
            oo = self.model(input_ids=torch.tensor([qids], device=self.dev),
                            position_ids=torch.arange(len(qids), device=self.dev)[None],
                            use_cache=False, output_hidden_states=True)
        qe = self._deanis(oo.hidden_states[self.lmid][0, -1:].float().cpu())[0]
        ss = self._seg_emb @ qe
        tokscore = torch.full((self.N,), -1e9)
        for s, (lo, hi) in enumerate(self.segments):
            tokscore[lo:hi] = ss[s]
        return qids, tokscore, int(ss.argmax())

    def answer(self, question, answer_token_str):
        qids, tokscore, best_seg = self._probe(question)
        order = torch.argsort(tokscore, descending=True)
        hot = sorted(set(order[:self.budget - self.recency].tolist()
                         + list(range(self.N - self.recency, self.N))))
        idx = torch.tensor(hot)
        for l in range(self.NL):
            if self.bits < 16:
                self._ansK[l] = self._dec(self._coldK[l][:, :, idx, :], self._coldKs[l][:, :, idx, :])
                self._ansV[l] = self._dec(self._coldV[l][:, :, idx, :], self._coldVs[l][:, :, idx, :])
            else:
                self._ansK[l] = self._coldK[l][:, :, idx, :].to(self.dev)
                self._ansV[l] = self._coldV[l][:, :, idx, :].to(self.dev)
        self._mode = "answer"
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([qids], device=self.dev),
                           position_ids=torch.arange(self.N, self.N + len(qids), device=self.dev)[None],
                           use_cache=False)
        logits = o.logits[0, -1].float()
        aid = self.tok.encode(answer_token_str, add_special_tokens=False)[0]
        rank = int((torch.argsort(logits, descending=True) == aid).nonzero()[0, 0])
        prob = float(torch.softmax(logits, -1)[aid])
        return rank, prob, best_seg

    # ---- persistence: cache as memory ----
    # Prefill once (e.g. over a wiki), save the cold store + catalog, reload next session and skip prefill.
    # The cache is SPECIFIC to this model + quantization + layer count; it is not portable across models.
    def save(self, path):
        torch.save({
            "version": 1,
            "model_name": getattr(self.model.config, "_name_or_path", "?"),
            "NL": self.NL, "NKV": self.NKV, "HD": self.HD, "hidden": self.hidden,
            "N": self.N, "segments": self.segments,
            "config": dict(win=self.win, budget=self.budget, recency=self.recency,
                           lmid=self.lmid, remove_pc=self.remove_pc, bits=self.bits),
            "coldK": self._coldK, "coldV": self._coldV,
            "coldKs": self._coldKs, "coldVs": self._coldVs,
            "mean": self._mean, "Vpc": self._Vpc, "seg_emb": self._seg_emb,
            "segvec": getattr(self, "_segvec", None),
        }, path)

    @classmethod
    def load(cls, path, model, tokenizer, device="cuda"):
        d = torch.load(path, map_location="cpu", weights_only=False)
        c = d["config"]
        r = cls(model, tokenizer, win=c["win"], budget=c["budget"], recency=c["recency"],
                lmid=c["lmid"], remove_pc=c["remove_pc"], bits=c["bits"], device=device)
        if r.NL != d["NL"] or r.HD != d["HD"]:
            raise ValueError(f"cache is for a different model ({d['model_name']}, NL={d['NL']}) — "
                             f"caches are model+quant specific and not portable")
        r.N = d["N"]; r.segments = d["segments"]
        r._coldK, r._coldV = d["coldK"], d["coldV"]
        r._coldKs, r._coldVs = d["coldKs"], d["coldVs"]
        r._mean, r._Vpc, r._seg_emb = d["mean"], d["Vpc"], d["seg_emb"]
        if d.get("segvec") is not None: r._segvec = d["segvec"]
        return r

    def cold_store_gb(self):
        per = self._coldK[0].element_size()
        n = sum(self._coldK[l].numel() + self._coldV[l].numel() for l in range(self.NL))
        scl = sum((self._coldKs[l].numel() + self._coldVs[l].numel()) * 2 for l in range(self.NL)) if self.bits < 16 else 0
        return (n * per + scl) / 1e9
