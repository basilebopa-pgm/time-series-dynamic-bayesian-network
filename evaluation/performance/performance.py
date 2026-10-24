import numpy as np
import pandas as pd

# Same four variables as the parameter-estimation comparison
COMMON_VARS = ["Volume", "Log_return", "Volatility", "Trend"]


def _n_slices(data):
    return len(set(c[1] for c in data.columns))


def crps_samples(samples, y):
    """
    Sample-based CRPS. samples: (N, S) predictive draws, y: (N,) observations.
    CRPS = E|X - y| - 0.5 E|X - X'|   (lower is better)
    """
    s = np.sort(samples, axis=1)
    S = s.shape[1]
    term1 = np.abs(s - y[:, None]).mean(axis=1)
    i = np.arange(1, S + 1)[None, :]
    term2 = (2.0 / S ** 2) * ((2 * i - S - 1) * s).sum(axis=1)   # E|X - X'|
    return term1 - 0.5 * term2


def forecast_metrics(model, data, context, variables=COMMON_VARS, n_steps=None,
                     n_samples=200, seed=0):
    """
    Forecast slices context..T-1 given slices 0..context-1.
    Returns a long DataFrame: variable, context, horizon (1-based) and
    rmse (of the predictive mean), mae (of the median), crps,
    cov90 (coverage of the central 90% interval), width90 (its mean width).
    """
    T = _n_slices(data)
    if not 1 <= context < T:
        raise ValueError(f"context must be in [1, {T - 1}].")
    n_steps = T - context if n_steps is None else min(n_steps, T - context)

    history = data.loc[:, [c for c in data.columns if c[1] < context]]
    samples, names = model.forecast(history, n_steps, n_samples=n_samples, seed=seed)

    rows = []
    for v in variables:
        if v not in names:
            continue
        j = names.index(v)
        for h in range(n_steps):
            y = data[(v, context + h)].to_numpy(dtype=float)
            s = samples[:, :, h, j].astype(float)
            q05, q50, q95 = np.quantile(s, [0.05, 0.5, 0.95], axis=1)
            rows.append({
                "variable": v, "context": context, "horizon": h + 1,
                "rmse": np.sqrt(np.mean((s.mean(axis=1) - y) ** 2)),
                "mae": np.mean(np.abs(q50 - y)),
                "crps": crps_samples(s, y).mean(),
                "cov90": np.mean((y >= q05) & (y <= q95)),
                "width90": np.mean(q95 - q05),
            })
    return pd.DataFrame(rows)


def persistence_metrics(data, context, variables=COMMON_VARS, n_steps=None):
    """Naive baseline: predict every future value as the last observed value."""
    T = _n_slices(data)
    n_steps = T - context if n_steps is None else min(n_steps, T - context)
    rows = []
    for v in variables:
        last = data[(v, context - 1)].to_numpy(dtype=float)
        for h in range(n_steps):
            y = data[(v, context + h)].to_numpy(dtype=float)
            rows.append({
                "variable": v, "context": context, "horizon": h + 1,
                "rmse": np.sqrt(np.mean((last - y) ** 2)),
                "mae": np.mean(np.abs(last - y)),
                "crps": np.nan, "cov90": np.nan, "width90": np.nan,
            })
    return pd.DataFrame(rows)


def cond_log_score(model, data, context):
    """
    Mean predictive log-density of the future slices given the first `context`
    slices, per future slice: (log p(x_0..x_{T-1}) - log p(x_0..x_{context-1})) / (T - context).
    Higher is better. Only comparable between models covering the same variables.
    """
    T = _n_slices(data)
    full = model.log_likelihood(data)
    ctx = model.log_likelihood(data.loc[:, [c for c in data.columns if c[1] < context]])
    return float(((full - ctx) / (T - context)).mean())