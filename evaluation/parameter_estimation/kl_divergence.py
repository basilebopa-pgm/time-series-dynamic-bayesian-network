import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal


def _joint_gaussian(dbn, n_time_slices, order):
    """
    Mean and covariance of the unrolled linear Gaussian DBN, computed directly
    from the CPDs (X_i = b0 + sum_j b_j * parent_j + noise), returned in `order`.
    `order` may be a subset of the variables, which gives the marginal.
    """
    bn = dbn._unroll(n_time_slices)
    cpds = {c.variable: c for c in bn.get_cpds()}
    topo = list(nx.topological_sort(bn))
    pos = {v: i for i, v in enumerate(topo)}
    n = len(topo)

    mean = np.zeros(n)
    cov = np.zeros((n, n))

    for v in topo:
        i = pos[v]
        c = cpds[v]
        beta = np.asarray(c.beta, dtype=float).ravel()
        parents = list(c.evidence or [])
        w = beta[1:]
        pidx = [pos[p] for p in parents]

        mean[i] = beta[0] + (w @ mean[pidx] if parents else 0.0)

        for k in range(i):
            cov[i, k] = cov[k, i] = (w @ cov[pidx, k]) if parents else 0.0

        var_parents = (w @ cov[np.ix_(pidx, pidx)] @ w) if parents else 0.0
        cov[i, i] = var_parents + float(c.std) ** 2

    idx = [pos[v] for v in order]
    return mean[idx], cov[np.ix_(idx, idx)]


def _check_variables(gt, model):
    """The model's variables must be a subset of the ground truth's."""
    extra = set(model._nodes()) - set(gt._nodes())
    if extra:
        raise ValueError(f"Model has variables the ground truth lacks: {sorted(extra)}")
    missing = sorted(set(gt._nodes()) - set(model._nodes()))
    return missing


def _gt_logp(gt, model, x, n_time_slices):
    """
    log p_gt(x) over the variables the model actually covers.
    If the model omits some GT variables, this is the log-density of the GT
    marginal over the model's variables (GT must be linear Gaussian).
    """
    missing = _check_variables(gt, model)
    if not missing:
        return gt.log_likelihood(x)

    if gt.model_type != "linear":
        raise NotImplementedError("Marginalising a neural ground truth is not supported.")

    nodes = sorted(model._nodes())
    cols = [(n, t) for t in range(n_time_slices) for n in nodes]
    names = [f"{n}_{t}" for n, t in cols]
    mean, cov = _joint_gaussian(gt, n_time_slices, names)
    return multivariate_normal(mean, cov, allow_singular=True).logpdf(x[cols].to_numpy(dtype=float))


def kl_monte_carlo(gt, model, n_time_slices, n_samples=20000, seed=0):
    """
    Monte Carlo estimate of KL(GT || model) = E_{x~GT}[log p_gt(x) - log p_model(x)].
    Returns (estimate, standard_error). Works for linear and neural models.
    If the model omits GT variables, the KL is taken against the GT marginal
    over the model's variables.
    """
    x = gt.simulate(n_samples=n_samples, n_time_slices=n_time_slices, seed=seed)
    diff = _gt_logp(gt, model, x, n_time_slices) - model.log_likelihood(x)
    return diff.mean(), diff.std() / np.sqrt(n_samples)


def kl_on_data(gt, model, data):
    """
    Same estimate, but using a held-out dataset sampled from the ground truth.
    Returns (estimate, standard_error).
    """
    T = len(set(c[1] for c in data.columns))
    diff = _gt_logp(gt, model, data, T) - model.log_likelihood(data)
    return diff.mean(), diff.std() / np.sqrt(len(data))


def kl_gaussian_exact(gt, model, n_time_slices):
    """
    Closed-form KL(GT || model) for linear Gaussian DBNs only.
    If the model omits GT variables, uses the GT marginal over the model's variables.
    """
    _check_variables(gt, model)
    nodes = sorted(model._nodes())
    order = [f"{n}_{t}" for t in range(n_time_slices) for n in nodes]
    mp, Sp = _joint_gaussian(gt, n_time_slices, order)
    mq, Sq = _joint_gaussian(model, n_time_slices, order)
    k = len(order)
    d = mq - mp
    Sq_inv = np.linalg.inv(Sq)
    return 0.5 * (
        np.trace(Sq_inv @ Sp) + d @ Sq_inv @ d - k
        + np.linalg.slogdet(Sq)[1] - np.linalg.slogdet(Sp)[1]
    )
