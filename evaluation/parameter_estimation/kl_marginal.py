import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

from evaluation.parameter_estimation.kl_divergence import _joint_gaussian

COMMON = ["Volume", "Log_return", "Volatility", "Trend"]


def _fast_kde_logpdf(samples, x, grid_size=4096):
    """
    Log-density at points x of a 1-D Gaussian KDE of `samples` (Scott bandwidth),
    computed by binning + Gaussian smoothing. Matches scipy.stats.gaussian_kde
    to ~1e-5 on average but is ~100x faster.
    """
    s = np.asarray(samples, dtype=float)
    n = len(s)
    bw = s.std(ddof=1) * n ** (-1.0 / 5.0)
    lo = min(s.min(), x.min()) - 10 * bw
    hi = max(s.max(), x.max()) + 10 * bw
    grid = np.linspace(lo, hi, grid_size)
    dx = grid[1] - grid[0]
    edges = np.append(grid - dx / 2, grid[-1] + dx / 2)
    hist, _ = np.histogram(s, bins=edges)
    dens = gaussian_filter1d(hist / (n * dx), sigma=bw / dx, mode="constant", truncate=10)
    return np.log(np.interp(x, grid, dens) + 1e-300)


def _check(model, variables):
    missing = set(variables) - set(model._nodes())
    if missing:
        raise ValueError(f"Model lacks variables: {sorted(missing)}")


def _names(variables, T):
    """Ordering used everywhere below: index = t * len(variables) + j."""
    return [f"{v}_{t}" for t in range(T) for v in variables]


def _gauss_kl(mp, Sp, mq, Sq):
    """Closed-form KL( N(mp, Sp) || N(mq, Sq) )."""
    k = len(mp)
    d = mq - mp
    Sq_inv = np.linalg.inv(Sq)
    return 0.5 * (
        np.trace(Sq_inv @ Sp) + d @ Sq_inv @ d - k
        + np.linalg.slogdet(Sq)[1] - np.linalg.slogdet(Sp)[1]
    )


def kl_per_variable(gt, model, variables, T, n_samples=10000, n_eval=5000, seed=0):
    """
    KL(GT || model) between the ONE-DIMENSIONAL marginals of each variable at
    each time slice. Returns a DataFrame (rows = variables, columns = slices).

    - GT is linear Gaussian, so its marginals are exact Gaussians.
    - Linear model: exact Gaussian marginals, closed-form KL.
    - Neural / LSTM model: the marginal is estimated by a kernel density
      estimate of model samples, and KL = E_{x~GT marginal}[log p - log q_kde].
    """
    _check(gt, variables)
    _check(model, variables)
    k = len(variables)
    mu_p, S_p = _joint_gaussian(gt, T, _names(variables, T))

    exact_model = model.model_type == "linear"
    if exact_model:
        mu_q, S_q = _joint_gaussian(model, T, _names(variables, T))
    else:
        samples = model.simulate(n_samples=n_samples, n_time_slices=T, seed=seed)

    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=variables, columns=range(T), dtype=float)

    for t in range(T):
        for j, v in enumerate(variables):
            i = t * k + j
            sp = np.sqrt(S_p[i, i])
            if exact_model:
                sq = np.sqrt(S_q[i, i])
                kl = (np.log(sq / sp)
                      + (sp ** 2 + (mu_p[i] - mu_q[i]) ** 2) / (2 * sq ** 2) - 0.5)
            else:
                x = rng.normal(mu_p[i], sp, size=n_eval)
                log_q = _fast_kde_logpdf(samples[(v, t)].to_numpy(dtype=float), x)
                kl = np.mean(norm.logpdf(x, mu_p[i], sp) - log_q)
            out.loc[v, t] = kl

    return out


def kl_joint_common(gt, model, variables, T, n_samples=50000, seed=0):
    """
    KL(GT || model) over the JOINT distribution of `variables` across T slices.

    - Linear model: exact (marginal of the model's joint Gaussian).
    - Neural / LSTM model: the model's marginal over `variables` is
      approximated by a Gaussian with the model's sample mean and covariance.
      This compares the first two moments only, so it can UNDERSTATE the true
      KL if the model's marginal is non-Gaussian. Treat it as a diagnostic.
    """
    _check(gt, variables)
    _check(model, variables)
    names = _names(variables, T)
    mp, Sp = _joint_gaussian(gt, T, names)

    if model.model_type == "linear":
        mq, Sq = _joint_gaussian(model, T, names)
    else:
        samples = model.simulate(n_samples=n_samples, n_time_slices=T, seed=seed)
        cols = [(v, t) for t in range(T) for v in variables]
        arr = samples[cols].to_numpy(dtype=float)
        mq = arr.mean(axis=0)
        Sq = np.cov(arr, rowvar=False)

    return _gauss_kl(mp, Sp, mq, Sq)