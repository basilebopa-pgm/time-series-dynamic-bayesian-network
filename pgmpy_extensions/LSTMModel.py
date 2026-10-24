import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSequenceModel(nn.Module):
    """
    Autoregressive LSTM density model for multivariate time series.

    At every time slice t the model outputs a full-covariance Gaussian over all
    variables, conditioned on the whole history x_0..x_{t-1}:

        p(x_t | x_<t) = N(mean_t, L_t L_t^T)

    L_t is a lower-triangular Cholesky factor, so correlations between
    variables inside a slice (e.g. Seasonality -> Log_return) are modelled.
    Data is standardised internally; log_likelihood() is in ORIGINAL units, so
    it is directly comparable with the DBN models and the ground truth.

    Data format: DataFrame with columns (variable, time_slice).
    """

    model_type = "lstm"

    def __init__(self, variables, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.variables = list(variables)
        self.d = len(self.variables)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # input = previous slice (d values) + "is first slice" flag
        self.lstm = nn.LSTM(
            self.d + 1, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mean_head = nn.Linear(hidden_dim, self.d)
        self.chol_head = nn.Linear(hidden_dim, self.d * (self.d + 1) // 2)

        self.register_buffer("tril_idx", torch.tril_indices(self.d, self.d))
        self.register_buffer("x_mean", torch.zeros(self.d))
        self.register_buffer("x_std", torch.ones(self.d))

        self.best_val_nll = None

    # ------------------------------------------------------------------
    # Interface expected by the KL code
    # ------------------------------------------------------------------
    def _nodes(self):
        return list(self.variables)

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def _to_array(self, data):
        slices = sorted(set(c[1] for c in data.columns))
        if slices != list(range(len(slices))):
            raise ValueError("Time slices must be consecutive integers starting from 0.")
        T = len(slices)
        cols = [(v, t) for t in range(T) for v in self.variables]
        missing = [c for c in cols if c not in set(data.columns)]
        if missing:
            raise ValueError(f"Missing data columns: {missing[:5]}")
        arr = data[cols].to_numpy(dtype=np.float32).reshape(len(data), T, self.d)
        return arr

    def _standardise(self, arr):
        return (torch.as_tensor(arr) - self.x_mean) / self.x_std

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    def _build_chol(self, raw):
        M = raw.new_zeros(*raw.shape[:-1], self.d, self.d)
        M[..., self.tril_idx[0], self.tril_idx[1]] = raw
        diag_raw = torch.diagonal(M, dim1=-2, dim2=-1)
        off = M - torch.diag_embed(diag_raw)
        diag = F.softplus(diag_raw) + 1e-4
        return off + torch.diag_embed(diag)

    def _inputs(self, z):
        B, T, _ = z.shape
        prev = torch.cat([z.new_zeros(B, 1, self.d), z[:, :-1]], dim=1)
        flag = z.new_zeros(B, T, 1)
        flag[:, 0] = 1.0
        return torch.cat([prev, flag], dim=-1)

    def _dist(self, inp, state=None):
        out, state = self.lstm(inp, state)
        mean = self.mean_head(out)
        L = self._build_chol(self.chol_head(out))
        return mean, L, state

    def _logp_std(self, z):
        """Log-density of whole sequences in standardised space, shape (B,)."""
        mean, L, _ = self._dist(self._inputs(z))
        r = (z - mean).unsqueeze(-1)
        w = torch.linalg.solve_triangular(L, r, upper=False).squeeze(-1)
        logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        lp = -0.5 * (w ** 2).sum(-1) - logdet - 0.5 * self.d * math.log(2 * math.pi)
        return lp.sum(dim=1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, data, epochs=300, lr=3e-3, batch_size=128, val_frac=0.15,
            patience=30, weight_decay=0.0, seed=None, verbose=True):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        arr = self._to_array(data)
        if not np.isfinite(arr).all():
            raise ValueError("Training data must be finite.")

        flat = arr.reshape(-1, self.d)
        with torch.no_grad():
            self.x_mean.copy_(torch.as_tensor(flat.mean(axis=0)))
            self.x_std.copy_(torch.as_tensor(flat.std(axis=0)).clamp_min(1e-6))

        z = self._standardise(arr)
        n = len(z)
        perm = torch.randperm(n)
        n_val = max(1, int(n * val_frac)) if val_frac > 0 else 0
        val_z = z[perm[:n_val]] if n_val else z[perm]
        tr_z = z[perm[n_val:]] if n_val else z[perm]

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

        best, best_state, bad = float("inf"), None, 0

        for epoch in range(epochs):
            self.train()
            idx = torch.randperm(len(tr_z))
            for i in range(0, len(tr_z), batch_size):
                batch = tr_z[idx[i:i + batch_size]]
                opt.zero_grad()
                loss = -self._logp_std(batch).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

            self.eval()
            with torch.no_grad():
                val = -self._logp_std(val_z).mean().item()
            sched.step(val)

            if val < best - 1e-4:
                best, best_state, bad = val, copy.deepcopy(self.state_dict()), 0
            else:
                bad += 1

            if verbose and (epoch % 10 == 0 or bad == 0):
                print(f"epoch {epoch:4d}  val NLL (std. space) {val:.4f}  best {best:.4f}")

            if bad >= patience:
                break

        self.load_state_dict(best_state)
        self.eval()
        self.best_val_nll = best
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_likelihood(self, data, chunk=4096):
        """Per-row log p(x_0, ..., x_{T-1}) in original units."""
        self.eval()
        arr = self._to_array(data)
        T = arr.shape[1]
        jac = T * torch.log(self.x_std).sum().item()
        out = []
        for i in range(0, len(arr), chunk):
            z = self._standardise(arr[i:i + chunk])
            out.append(self._logp_std(z).numpy().astype(np.float64))
        return np.concatenate(out) - jac

    @torch.no_grad()
    def simulate(self, n_samples=10, n_time_slices=2, seed=None):
        """Ancestral sampling. Returns a DataFrame with (variable, t) columns."""
        self.eval()
        gen = torch.Generator().manual_seed(0 if seed is None else int(seed))
        prev = torch.zeros(n_samples, self.d)
        state, out = None, []

        for t in range(n_time_slices):
            flag = torch.full((n_samples, 1), 1.0 if t == 0 else 0.0)
            inp = torch.cat([prev, flag], dim=-1).unsqueeze(1)
            mean, L, state = self._dist(inp, state)
            eps = torch.randn(n_samples, self.d, generator=gen)
            z = mean[:, 0] + (L[:, 0] @ eps.unsqueeze(-1)).squeeze(-1)
            out.append(z)
            prev = z

        arr = (torch.stack(out, dim=1) * self.x_std + self.x_mean).numpy()
        cols = [(v, t) for t in range(n_time_slices) for v in self.variables]
        df = pd.DataFrame(arr.reshape(n_samples, -1), columns=cols)
        df.columns = pd.MultiIndex.from_tuples(cols)
        return df

    @torch.no_grad()
    def forecast(self, history, n_steps, n_samples=200, seed=None):
        """
        Sample the future given observed history (columns (variable, t), t < k).
        Returns (samples, variables); samples has shape
        (n_rows, n_samples, n_steps, n_variables).
        """
        self.eval()
        gen = torch.Generator().manual_seed(0 if seed is None else int(seed))

        arr = self._to_array(history)                    # (N, k, d)
        N = arr.shape[0]
        z = self._standardise(arr)

        # Run the history; the last step's output is the distribution of slice k.
        padded = torch.cat([z, z.new_zeros(N, 1, self.d)], dim=1)
        out, state = self.lstm(self._inputs(padded))
        cur = out[:, -1].repeat_interleave(n_samples, dim=0)
        state = tuple(s.repeat_interleave(n_samples, dim=1) for s in state)

        B = N * n_samples
        samples = []
        for _ in range(n_steps):
            mean = self.mean_head(cur)
            L = self._build_chol(self.chol_head(cur))
            eps = torch.randn(B, self.d, generator=gen)
            zs = mean + (L @ eps.unsqueeze(-1)).squeeze(-1)
            samples.append(zs)
            inp = torch.cat([zs, zs.new_zeros(B, 1)], dim=-1).unsqueeze(1)
            step_out, state = self.lstm(inp, state)
            cur = step_out[:, 0]

        res = torch.stack(samples, dim=1).reshape(N, n_samples, n_steps, self.d)
        res = res * self.x_std + self.x_mean
        return res.numpy(), list(self.variables)