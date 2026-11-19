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

Phase 2 additions (all OFF by default, so Phase 1 configs reproduce bit-for-bit):
  * seasonal_lags: extra inputs y_{t-k} (k in e.g. [7, 14, 28]) + observed flags; in ancestral sampling the
    lag comes from history if t-k <= origin, else from the sample path itself
  * age_feature: log1p(days since the series' first available day), computed from history only
  * validation: the last `horizon` days of HISTORY are held out as validation targets (training windows end
    before them); early stopping on validation NLL with best-checkpoint restore. The test window is never used.
  * optimizer: adam (Phase 1, coupled L2) | adamw (decoupled weight decay, Loshchilov & Hutter 2019)
  * per-path NB parameters are kept after predict (`last_dist`) -> Rao-Blackwellised P(Y=0) and exact
    positive-part sampling for the hurdle hybrid (models/hybrid.py); samples/curves/checkpoints exported
"""
from __future__ import annotations

import math
import time
from typing import Any, ClassVar

import numpy as np

from forecastkit.utils import log, register_model

from .base import BaseForecaster

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError("torch is required for the deepar model: pip install torch") from e


class _Net(nn.Module):
    def __init__(self, n_cov, cards, emb_dims, hidden, layers, dropout, likelihood, n_extra=0):
        super().__init__()
        self.likelihood = likelihood
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, emb_dims)])
        in_dim = 3 + n_extra + n_cov + sum(emb_dims)
        self.rnn = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, 3 if likelihood == "studentt" else 2)

    def static_vec(self, cats):
        if not self.embs:
            return cats.new_zeros((cats.shape[0], 0), dtype=torch.float32)
        return torch.cat([e(cats[:, i]) for i, e in enumerate(self.embs)], dim=1)

    def forward(self, lag, flag, cov, cats, logscale, state=None, extra=None):
        B, L = lag.shape
        sv = self.static_vec(cats).unsqueeze(1).expand(B, L, -1)
        parts = [lag.unsqueeze(-1), flag.unsqueeze(-1), logscale.view(B, 1, 1).expand(B, L, 1)]
        if extra is not None:
            parts.append(extra)
        x = torch.cat([*parts, cov, sv], -1)
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
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "likelihood": "negbin", "context_length": 56, "hidden": 40, "layers": 2, "dropout": 0.1, "steps": 600,
        "finetune_steps": 250, "batch_size": 128, "lr": 1e-3, "weight_decay": 1e-5, "num_samples": 100,
        "use_item_id": True, "item_dropout": 0.15, "emb_dim": 8, "use_mask": True, "warm_start": True,
        "refit_every": 1, "train_span": 730, "include_covariates": None, "exclude_covariates": None,
        "num_threads": 1, "infer_chunk": 128, "clip": 10.0, "device": "auto",
        # Phase 2 (defaults = Phase 1 behaviour)
        "seasonal_lags": [], "age_feature": False, "optimizer": "adam", "validation": False,
        "eval_every": 100, "patience": 0, "restore_best": True, "export": True,
    }

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
        try:
            torch.set_num_threads(int(self.pp("num_threads")))
        except (RuntimeError, ValueError):
            pass
        self.dev = self._device()
        cov = ctx.covariate_idx(self.pp("include_covariates"), self.pp("exclude_covariates"))
        self.cov_idx = cov
        Kh = np.ascontiguousarray(ctx.known_hist[:, :, cov], dtype=np.float32)
        Kf = np.ascontiguousarray(ctx.known_fut[:, :, cov], dtype=np.float32)
        if not hasattr(self, "cov_mu"):  # frozen at first fit (data <= first origin) -> warm starts stay consistent
            flat = Kh.reshape(-1, Kh.shape[2])
            self.cov_mu = flat.mean(0)
            self.cov_sd = flat.std(0) + 1e-6
        self.Kh = torch.tensor((Kh - self.cov_mu) / self.cov_sd, dtype=torch.float32, device=self.dev)
        self.Kf = torch.tensor((Kf - self.cov_mu) / self.cov_sd, dtype=torch.float32, device=self.dev)
        y = np.ascontiguousarray(ctx.target(self.pp("use_mask")), dtype=np.float32)
        self.Y = torch.tensor(y, dtype=torch.float32, device=self.dev)
        self.cats = torch.tensor(self._vocab(ctx), dtype=torch.long, device=self.dev)
        self.lags = [int(k) for k in (self.pp("seasonal_lags") or [])]
        self.use_age = bool(self.pp("age_feature"))
        if self.use_age:
            a_hist, a_fut = self.age_arrays(ctx.available, ctx.horizon)
            self.age_h = torch.tensor(a_hist, dtype=torch.float32, device=self.dev)
            self.age_f = torch.tensor(a_fut, dtype=torch.float32, device=self.dev)

    @staticmethod
    def age_arrays(available: np.ndarray, H: int):
        """Causal product-age feature log1p(days since first available day) / 7 for history and the future.
        Uses only the availability mask of the history slice (first availability = launch after leading zeros)."""
        N, Th = available.shape
        first = np.where(available.any(1), available.argmax(1), Th)
        t = np.arange(Th + H)[None, :]
        age = np.maximum(t - first[:, None], 0).astype(np.float32)
        f = np.log1p(age) / 7.0
        return f[:, :Th], f[:, Th:]

    @property
    def n_extra(self):
        return 2 * len(self.lags) + int(self.use_age)

    def _build(self, ctx):
        cards = [len(self.vocab[c]) + 1 for c in self.cat_cols]
        dims = [int(self.pp("emb_dim")) if c == (ctx.meta.get("item_col")) else max(2, min(4, int(self.pp("emb_dim")) // 2))
                for c in self.cat_cols]
        return _Net(len(self.cov_idx), cards, dims, int(self.pp("hidden")), int(self.pp("layers")),
                    float(self.pp("dropout")), self.pp("likelihood"), n_extra=self.n_extra).to(self.dev)

    def _extra_train(self, i_exp, pos, scale):
        """[B, L, n_extra] seasonal lags y_{p-k} (scaled, + observed flag) and age at positions p, or None."""
        if not self.n_extra:
            return None
        parts = []
        for k in self.lags:
            j = pos - k
            v = self.Y[i_exp, j.clamp(min=0)]
            v = torch.where(j >= 0, v, torch.full_like(v, float("nan")))
            parts += [torch.nan_to_num(v, nan=0.0) / scale[:, None], (~torch.isnan(v)).float()]
        if self.use_age:
            parts.append(self.age_h[i_exp, pos])
        return torch.stack(parts, -1)

    # ------------------------------------------------------------------ training
    def _batch(self, B, ctx_len, L, g, hold_out=0):
        N, Th = self.Y.shape
        hi = Th - L - hold_out                       # last target of a training window <= Th-1-hold_out
        lo = max(0, hi - int(self.pp("train_span")))
        if hi < 0:
            return None
        out = None
        for _ in range(20):
            n = B * 3
            i = torch.randint(0, N, (n,), generator=g)
            s = torch.randint(lo, hi + 1, (n,), generator=g)
            i, s = i.to(self.dev), s.to(self.dev)
            pos = s[:, None] + torch.arange(L, device=self.dev)[None, :]
            i_exp = i[:, None].expand_as(pos)
            tgt = self.Y[i_exp, pos]
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

    def _window_loss(self, net, i, pos, tgt, ctx_len, Ypad, cats, loss_from=0):
        """Masked mean NLL of the windows; loss counted on positions >= loss_from only."""
        i_exp = i[:, None].expand_as(pos)
        lag = Ypad[i_exp, pos]
        cov = self.Kh[i_exp, pos]
        ctxv = tgt[:, :ctx_len]
        m = ~torch.isnan(ctxv)
        scale = 1.0 + torch.where(m, ctxv, torch.zeros_like(ctxv)).sum(1) / m.sum(1).clamp(min=1)
        flag = (~torch.isnan(lag)).float()
        lag_in = torch.nan_to_num(lag, nan=0.0) / scale[:, None]
        raw, _ = net(lag_in, flag, cov, cats, scale.log(), extra=self._extra_train(i_exp, pos, scale))
        params = net.dist_params(raw, scale)
        tmask = ~torch.isnan(tgt)
        if loss_from:
            tmask[:, :loss_from] = False
        nll = net.nll(params, torch.nan_to_num(tgt, nan=0.0))
        return (nll * tmask).sum(), tmask.sum()

    @torch.no_grad()
    def _val_loss(self, net, ctx_len, H, Ypad):
        """Teacher-forced NLL on the last H days of HISTORY (all series), windows ending at the origin."""
        N, Th = self.Y.shape
        L = ctx_len + H
        if Th < L:
            return float("nan")
        was = net.training
        net.eval()
        pos_row = torch.arange(Th - L, Th, device=self.dev)
        tot, cnt = 0.0, 0
        cs = int(self.pp("infer_chunk")) * 4
        for a in range(0, N, cs):
            i = torch.arange(a, min(a + cs, N), device=self.dev)
            pos = pos_row[None, :].expand(len(i), L)
            tgt = self.Y[i[:, None].expand_as(pos), pos]
            s, c = self._window_loss(net, i, pos, tgt, ctx_len, Ypad, self.cats[i], loss_from=ctx_len)
            tot, cnt = tot + float(s), cnt + int(c)
        net.train(was)
        return tot / max(cnt, 1)

    def _fit_loop(self, net, steps, ctx_len, H, seed, lr, warm=False):
        L = ctx_len + H
        g = torch.Generator().manual_seed(seed)
        wd = float(self.pp("weight_decay"))
        if self.pp("optimizer") == "adamw":
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
        Ypad = torch.cat([torch.full((self.Y.shape[0], 1), float("nan"), device=self.dev), self.Y], 1)  # Ypad[:,t] = y_{t-1}
        use_val = bool(self.pp("validation"))
        hold = H if use_val else 0
        every, patience = int(self.pp("eval_every")), int(self.pp("patience"))
        hist = {"train_step": [], "train_loss": [], "val_step": [], "val_loss": [], "warm": bool(warm), "lr": lr,
                "max_steps": steps, "best_step": None, "best_val": None, "stopped_step": steps}
        best, best_state, bad = float("inf"), None, 0

        def check(step):
            nonlocal best, best_state, bad
            v = self._val_loss(net, ctx_len, H, Ypad)
            hist["val_step"].append(step)
            hist["val_loss"].append(v)
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                hist["best_step"], hist["best_val"] = step, v
            else:
                bad += 1
            return bool(patience) and bad >= patience

        net.train()
        losses = []
        if use_val and warm:
            check(0)                                 # fine-tuning may not help: the incoming weights compete too
        for step in range(steps):
            b = self._batch(int(self.pp("batch_size")), ctx_len, L, g, hold_out=hold)
            if b is None:
                break
            i, pos, tgt = b
            cats = self.cats[i].clone()
            if self.item_slot is not None and self.pp("item_dropout") > 0:
                drop = (torch.rand(len(i), generator=g) < float(self.pp("item_dropout"))).to(self.dev)
                cats[drop, self.item_slot] = 0
            s, c = self._window_loss(net, i, pos, tgt, ctx_len, Ypad, cats)
            loss = s / c.clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), float(self.pp("clip")))
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            if use_val and ((step + 1) % every == 0 or step + 1 == steps):
                if check(step + 1):
                    hist["stopped_step"] = step + 1
                    break
        if use_val and best_state is not None and bool(self.pp("restore_best")):
            net.load_state_dict(best_state)
        k = max(1, every // 4)
        hist["train_step"] = list(range(k, len(losses) + 1, k))
        hist["train_loss"] = [float(np.mean(losses[j - k:j])) for j in hist["train_step"]]
        self.last_history = hist
        net.eval()
        return losses

    def fit(self, ctx):
        self.ctx = ctx
        self.fits = getattr(self, "fits", 0)
        self._prepare(ctx)
        H, ctx_len = ctx.horizon, int(self.pp("context_length"))
        need_train = (self.fits % int(self.pp("refit_every")) == 0) or not hasattr(self, "net")
        self.last_history = None
        if need_train:
            warm = bool(self.pp("warm_start")) and hasattr(self, "net")
            if not warm:
                torch.manual_seed(int(ctx.seed))
                self.net = self._build(ctx)
            steps = int(self.pp("finetune_steps")) if warm else int(self.pp("steps"))
            lr = float(self.pp("lr")) * (0.5 if warm else 1.0)
            t0 = time.time()
            self.losses = self._fit_loop(self.net, steps, ctx_len, H, int(ctx.seed) + 17 * self.fits, lr, warm=warm)
            if self.last_history is not None:
                self.last_history["seconds"] = time.time() - t0
                self.last_history["origin"] = str(ctx.origin.date())
            log.debug("deepar %s %d steps in %.1fs, final loss %.4f", "fine-tune" if warm else "train", steps,
                      time.time() - t0, np.mean(self.losses[-20:]) if self.losses else float("nan"))
        self.fits += 1
        return self

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample_paths(self):
        """Ancestral sampling. Returns samples [N,S,H] and, for negbin, per-path parameters mu/alpha [N,S,H]."""
        ctx, net = self.ctx, self.net
        N, H, Th = ctx.N, ctx.horizon, ctx.Th
        ctx_len, S = int(self.pp("context_length")), int(self.pp("num_samples"))
        torch.manual_seed(int(ctx.seed) + 999)
        Ypad = torch.cat([torch.full((N, 1), float("nan"), device=self.dev), self.Y], 1)
        samples = np.zeros((N, S, H), dtype=np.float32)
        nb = net.likelihood == "negbin"
        MU = np.zeros((N, S, H), dtype=np.float32) if nb else None
        AL = np.zeros((N, S, H), dtype=np.float32) if nb else None
        cs = int(self.pp("infer_chunk"))
        pos = torch.arange(max(Th - ctx_len, 0), Th, device=self.dev)
        maxlag = max(self.lags) if self.lags else 0
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
            pos_b = pos[None, :].expand(B, -1)
            ex = self._extra_train(idx[:, None].expand_as(pos_b), pos_b, scale)
            _, state = net(lag_in, flag, self.Kh[idx][:, pos], cats, scale.log(), extra=ex)
            # expand to S samples
            state = tuple(s.repeat_interleave(S, dim=1) for s in state)
            cats_s = cats.repeat_interleave(S, 0)
            scale_s = scale.repeat_interleave(S)
            if maxlag:
                tail = torch.full((B, maxlag), float("nan"), device=self.dev)
                k0 = min(maxlag, Th)
                tail[:, maxlag - k0:] = self.Y[idx][:, Th - k0:Th]
                tail_s = tail.repeat_interleave(S, 0)     # position Th - maxlag + j
            if self.use_age:
                age_s = self.age_f[idx].repeat_interleave(S, 0)
            last = self.Y[idx][:, Th - 1]
            prev = last.repeat_interleave(S)               # true y_{T-1} (may be NaN if masked)
            out = torch.zeros(B * S, H, device=self.dev)
            mu_o = torch.zeros(B * S, H, device=self.dev) if nb else None
            al_o = torch.zeros(B * S, H, device=self.dev) if nb else None
            for h in range(H):
                flag_h = (~torch.isnan(prev)).float()
                lag_h = (torch.nan_to_num(prev, nan=0.0) / scale_s).unsqueeze(1)
                cov_h = self.Kf[idx][:, h:h + 1].repeat_interleave(S, 0)
                ex_h = None
                if self.n_extra:
                    parts = []
                    for k in self.lags:
                        v = out[:, h - k] if h - k >= 0 else tail_s[:, maxlag + h - k]
                        parts += [torch.nan_to_num(v, nan=0.0) / scale_s, (~torch.isnan(v)).float()]
                    if self.use_age:
                        parts.append(age_s[:, h])
                    ex_h = torch.stack(parts, -1).unsqueeze(1)
                raw, state = net(lag_h, flag_h.unsqueeze(1), cov_h, cats_s, scale_s.log(), state, extra=ex_h)
                params = net.dist_params(raw, scale_s)
                z = net.sample(params)[:, 0]
                out[:, h] = z
                if nb:
                    mu_o[:, h], al_o[:, h] = params[0][:, 0], params[1][:, 0]
                prev = z
            samples[a:a + B] = out.view(B, S, H).cpu().numpy()
            if nb:
                MU[a:a + B] = mu_o.view(B, S, H).cpu().numpy()
                AL[a:a + B] = al_o.view(B, S, H).cpu().numpy()
        return samples, ({"mu": MU, "alpha": AL} if nb else None)

    @staticmethod
    def nb_p0(mu, alpha):
        """P(Y=0) under NB(mean mu, dispersion alpha): (r/(r+mu))^r with r = 1/alpha."""
        mu, alpha = np.asarray(mu, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
        r = 1.0 / alpha
        return np.exp(r * (np.log(r) - np.log(r + mu)))

    def zero_probability(self):
        """[N,H] P(Y=0) of the predictive marginal: Rao-Blackwellised over sample paths for negbin
        (mean of per-path NB zero masses), otherwise the sample frequency."""
        if self.last_dist is not None:
            return self.nb_p0(self.last_dist["mu"], self.last_dist["alpha"]).mean(1)
        return (self.last_samples == 0).mean(1)

    @torch.no_grad()
    def predict(self, quantiles):
        samples, dist = self.sample_paths()
        self.last_samples, self.last_dist = samples, dist
        qs = np.asarray(quantiles, dtype=float)
        Q = np.quantile(samples, qs, axis=1).transpose(1, 2, 0)  # [N,H,Q]
        return self.finalize(Q, mean=samples.mean(1))

    # ------------------------------------------------------------------ artifacts
    def export(self, out_dir, ctx):
        """Per-origin artifacts: predictive samples + P(Y=0) (for PIT / zero calibration), training curves, weights."""
        if not bool(self.pp("export")):
            return
        import json
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = str(ctx.origin.date())
        np.savez_compressed(out / f"pred_{tag}.npz", samples=self.last_samples.astype(np.float32),
                            p0=self.zero_probability().astype(np.float32), series_ids=ctx.series_ids)
        if self.last_history is not None:
            (out / f"train_{tag}.json").write_text(json.dumps(self.last_history))
        torch.save(self.net.state_dict(), out / f"weights_{tag}.pt")
