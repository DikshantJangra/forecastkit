"""DeepAR (Salinas et al., 2020) - global autoregressive RNN with a probabilistic output, in plain PyTorch
(no GluonTS dependency, so it runs wherever torch does and plugs into the same interface).

  * likelihood: negative binomial (counts, over-dispersion) | gaussian | student-t   (ablation switch)
  * static categorical embeddings (hierarchy + optional item id) + dynamic known-future covariates
  * per-window mean scaling  nu = 1 + mean(context)  (the DeepAR scale trick)
  * loss is MASKED on censored / unavailable days (use_mask=True) -> stockouts are not learned as demand
  * cold start: item-id embedding index 0 = 'unknown item'; trained with item dropout, used for cold series
  * inference: ancestral sampling of S sample paths -> quantiles + mean from samples
  * warm start across rolling origins (fine-tune instead of retraining from scratch)
Windows are sampled only from history <= origin, so training never sees the forecast window.
"""
from __future__ import annotations

import math
import time

import numpy as np

from forecastkit.schema import QuantileForecast
from forecastkit.utils import log, register_model
from .base import BaseForecaster

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("torch is required for the deepar model: pip install torch") from e


class _Net(nn.Module):
    def __init__(self, n_cov, cards, emb_dims, hidden, layers, dropout, likelihood):
        super().__init__()
        self.likelihood = likelihood
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, emb_dims)])
        in_dim = 3 + n_cov + sum(emb_dims)
        self.rnn = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, 3 if likelihood == "studentt" else 2)

    def static_vec(self, cats):
        if not self.embs:
            return cats.new_zeros((cats.shape[0], 0), dtype=torch.float32)
        return torch.cat([e(cats[:, i]) for i, e in enumerate(self.embs)], dim=1)

    def forward(self, lag, flag, cov, cats, logscale, state=None):
        B, L = lag.shape
        sv = self.static_vec(cats).unsqueeze(1).expand(B, L, -1)
        x = torch.cat([lag.unsqueeze(-1), flag.unsqueeze(-1), logscale.view(B, 1, 1).expand(B, L, 1), cov, sv], -1)
        h, state = self.rnn(x, state)
        return self.out(h), state

    # -------- distribution ---------------------------------------------------
    def dist_params(self, raw, scale):
        s = scale.view(-1, 1)
        if self.likelihood == "negbin":
            return F.softplus(raw[..., 0]) * s + 1e-4, F.softplus(raw[..., 1]) / s.sqrt() + 1e-4
        if self.likelihood == "gaussian":
            return raw[..., 0] * s, F.softplus(raw[..., 1]) * s + 1e-3
        return raw[..., 0] * s, F.softplus(raw[..., 1]) * s + 1e-3, 2.0 + F.softplus(raw[..., 2])

    def nll(self, params, y):
        if self.likelihood == "negbin":
            mu, alpha = params
            r = 1.0 / alpha
            lp = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
                  + r * (torch.log(r) - torch.log(r + mu)) + y * (torch.log(mu) - torch.log(r + mu)))
            return -lp
        if self.likelihood == "gaussian":
            mu, sd = params
            return 0.5 * math.log(2 * math.pi) + torch.log(sd) + 0.5 * ((y - mu) / sd) ** 2
        mu, sd, df = params
        z = (y - mu) / sd
        lp = (torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2) - 0.5 * torch.log(df * math.pi) - torch.log(sd)
              - (df + 1) / 2 * torch.log1p(z * z / df))
        return -lp

    def sample(self, params):
        if self.likelihood == "negbin":
            mu, alpha = params
            r = 1.0 / alpha
            lam = torch.distributions.Gamma(r, r / mu).sample()
            return torch.poisson(lam)
        if self.likelihood == "gaussian":
            mu, sd = params
            return torch.clamp(mu + sd * torch.randn_like(mu), min=0.0)
        mu, sd, df = params
        t = torch.distributions.StudentT(df).sample()
        return torch.clamp(mu + sd * t, min=0.0)


@register_model("deepar")
class DeepARForecaster(BaseForecaster):
    DEFAULTS = dict(likelihood="negbin", context_length=56, hidden=40, layers=2, dropout=0.1, steps=600,
                    finetune_steps=250, batch_size=128, lr=1e-3, weight_decay=1e-5, num_samples=100,
                    use_item_id=True, item_dropout=0.15, emb_dim=8, use_mask=True, warm_start=True,
                    refit_every=1, train_span=730, include_covariates=None, exclude_covariates=None,
                    num_threads=2, infer_chunk=128, clip=10.0, device="auto")

    def pp(self, k):
        return self.params.get(k, self.DEFAULTS[k])

    # ------------------------------------------------------------------ setup
    def _vocab(self, ctx):
        cols = list(ctx.meta.get("static_cat_cols") or [])
        item_col = ctx.meta.get("item_col") if self.pp("use_item_id") else None
        self.cat_cols = cols + ([item_col] if item_col else [])
        self.item_slot = len(cols) if item_col else None
        if not hasattr(self, "vocab"):
            self.vocab = {}
            for c in self.cat_cols:
                levels = sorted(ctx.static[c].astype(str).unique())
                self.vocab[c] = {v: i + 1 for i, v in enumerate(levels)}  # 0 reserved: 'unknown'
        cats = np.zeros((ctx.N, len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            cats[:, j] = ctx.static[c].astype(str).map(self.vocab[c]).fillna(0).astype(int).to_numpy()
        if self.item_slot is not None:
            cats[ctx.cold, self.item_slot] = 0   # cold-start items: unknown-item embedding
        return cats

    def _device(self):
        want = self.pp("device")
        if want == "auto":
            want = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(want)

    def _prepare(self, ctx):
        torch.set_num_threads(int(self.pp("num_threads")))
        self.dev = self._device()
        cov = ctx.covariate_idx(self.pp("include_covariates"), self.pp("exclude_covariates"))
        self.cov_idx = cov
        Kh = ctx.known_hist[:, :, cov].astype(np.float32)
        Kf = ctx.known_fut[:, :, cov].astype(np.float32)
        if not hasattr(self, "cov_mu"):  # frozen at first fit (data <= first origin) -> warm starts stay consistent
            flat = Kh.reshape(-1, Kh.shape[2])
            self.cov_mu = flat.mean(0)
            self.cov_sd = flat.std(0) + 1e-6
        self.Kh = torch.from_numpy((Kh - self.cov_mu) / self.cov_sd).to(self.dev)
        self.Kf = torch.from_numpy((Kf - self.cov_mu) / self.cov_sd).to(self.dev)
        y = ctx.target(self.pp("use_mask")).astype(np.float32)
        self.Y = torch.from_numpy(y).to(self.dev)
        self.cats = torch.from_numpy(self._vocab(ctx)).to(self.dev)

    def _build(self, ctx):
        cards = [len(self.vocab[c]) + 1 for c in self.cat_cols]
        dims = [int(self.pp("emb_dim")) if c == (ctx.meta.get("item_col")) else max(2, min(4, int(self.pp("emb_dim")) // 2))
                for c in self.cat_cols]
        return _Net(len(self.cov_idx), cards, dims, int(self.pp("hidden")), int(self.pp("layers")),
                    float(self.pp("dropout")), self.pp("likelihood")).to(self.dev)

    # ------------------------------------------------------------------ training
    def _batch(self, B, ctx_len, L, g):
        N, Th = self.Y.shape
        lo = max(0, Th - L - int(self.pp("train_span")))
        hi = Th - L
        out = None
        for _ in range(20):
            n = B * 3
            i = torch.randint(0, N, (n,), generator=g)
            s = torch.randint(lo, hi + 1, (n,), generator=g)
            i, s = i.to(self.dev), s.to(self.dev)
            pos = s[:, None] + torch.arange(L, device=self.dev)[None, :]
            tgt = self.Y[i[:, None], pos]
            ok = (~torch.isnan(tgt[:, ctx_len:])).sum(1) >= 1
            if ok.sum() == 0:
                continue
            sel = torch.nonzero(ok).squeeze(1)[:B]
            out = (i[sel], pos[sel], tgt[sel])
            break
        if out is None:
            return None
        i, pos, tgt = out
        return i, pos, tgt

    def _fit_loop(self, net, steps, ctx_len, H, seed, lr):
        L = ctx_len + H
        g = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=float(self.pp("weight_decay")))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
        Ypad = torch.cat([torch.full((self.Y.shape[0], 1), float("nan"), device=self.dev), self.Y], 1)  # Ypad[:,t] = y_{t-1}
        net.train()
        losses = []
        for step in range(steps):
            b = self._batch(int(self.pp("batch_size")), ctx_len, L, g)
            if b is None:
                break
            i, pos, tgt = b
            lag = Ypad[i[:, None], pos]
            cov = self.Kh[i[:, None], pos]
            cats = self.cats[i].clone()
            if self.item_slot is not None and self.pp("item_dropout") > 0:
                drop = (torch.rand(len(i), generator=g) < float(self.pp("item_dropout"))).to(self.dev)
                cats[drop, self.item_slot] = 0
            ctxv = tgt[:, :ctx_len]
            m = ~torch.isnan(ctxv)
            scale = 1.0 + torch.where(m, ctxv, torch.zeros_like(ctxv)).sum(1) / m.sum(1).clamp(min=1)
            flag = (~torch.isnan(lag)).float()
            lag_in = torch.nan_to_num(lag, nan=0.0) / scale[:, None]
            raw, _ = net(lag_in, flag, cov, cats, scale.log())
            params = net.dist_params(raw, scale)
            tmask = ~torch.isnan(tgt)
            nll = net.nll(params, torch.nan_to_num(tgt, nan=0.0))
            loss = (nll * tmask).sum() / tmask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), float(self.pp("clip")))
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        net.eval()
        return losses

    def fit(self, ctx):
        self.ctx = ctx
        self.fits = getattr(self, "fits", 0)
        self._prepare(ctx)
        H, ctx_len = ctx.horizon, int(self.pp("context_length"))
        need_train = (self.fits % int(self.pp("refit_every")) == 0) or not hasattr(self, "net")
        if need_train:
            warm = bool(self.pp("warm_start")) and hasattr(self, "net")
            if not warm:
                torch.manual_seed(int(ctx.seed))
                self.net = self._build(ctx)
            steps = int(self.pp("finetune_steps")) if warm else int(self.pp("steps"))
            lr = float(self.pp("lr")) * (0.5 if warm else 1.0)
            t0 = time.time()
            self.losses = self._fit_loop(self.net, steps, ctx_len, H, int(ctx.seed) + 17 * self.fits, lr)
            log.debug("deepar %s %d steps in %.1fs, final loss %.4f", "fine-tune" if warm else "train", steps,
                      time.time() - t0, np.mean(self.losses[-20:]) if self.losses else float("nan"))
        self.fits += 1
        return self

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, quantiles):
        ctx, net = self.ctx, self.net
        N, H, Th = ctx.N, ctx.horizon, ctx.Th
        ctx_len, S = int(self.pp("context_length")), int(self.pp("num_samples"))
        torch.manual_seed(int(ctx.seed) + 999)
        Ypad = torch.cat([torch.full((N, 1), float("nan"), device=self.dev), self.Y], 1)
        samples = np.zeros((N, S, H), dtype=np.float32)
        cs = int(self.pp("infer_chunk"))
        pos = torch.arange(max(Th - ctx_len, 0), Th, device=self.dev)
        for a in range(0, N, cs):
            idx = torch.arange(a, min(a + cs, N), device=self.dev)
            B = len(idx)
            tgt = self.Y[idx][:, pos]
            m = ~torch.isnan(tgt)
            scale = 1.0 + torch.where(m, tgt, torch.zeros_like(tgt)).sum(1) / m.sum(1).clamp(min=1)
            lag = Ypad[idx][:, pos]
            flag = (~torch.isnan(lag)).float()
            lag_in = torch.nan_to_num(lag, nan=0.0) / scale[:, None]
            cats = self.cats[idx]
            _, state = net(lag_in, flag, self.Kh[idx][:, pos], cats, scale.log())
            # expand to S samples
            state = tuple(s.repeat_interleave(S, dim=1) for s in state)
            cats_s = cats.repeat_interleave(S, 0)
            scale_s = scale.repeat_interleave(S)
            last = self.Y[idx][:, Th - 1]
            prev = last.repeat_interleave(S)               # true y_{T-1} (may be NaN if masked)
            out = torch.zeros(B * S, H, device=self.dev)
            for h in range(H):
                flag_h = (~torch.isnan(prev)).float()
                lag_h = (torch.nan_to_num(prev, nan=0.0) / scale_s).unsqueeze(1)
                cov_h = self.Kf[idx][:, h:h + 1].repeat_interleave(S, 0)
                raw, state = net(lag_h, flag_h.unsqueeze(1), cov_h, cats_s, scale_s.log(), state)
                params = net.dist_params(raw, scale_s)
                z = net.sample(params)[:, 0]
                out[:, h] = z
                prev = z
            samples[a:a + B] = out.view(B, S, H).cpu().numpy()
        qs = np.asarray(quantiles, dtype=float)
        Q = np.quantile(samples, qs, axis=1).transpose(1, 2, 0)  # [N,H,Q]
        return self.finalize(Q, mean=samples.mean(1))
