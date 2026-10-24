import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.Simulation.GTM_DBN import create_ground_truth
from evaluation.parameter_estimation.kl_divergence import (
    kl_monte_carlo, kl_on_data, kl_gaussian_exact, _gt_logp
)
from evaluation.parameter_estimation.kl_marginal import (
    kl_per_variable, kl_joint_common, COMMON
)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================================
# 1. Load everything
# =========================================================================
ground_truth = create_ground_truth()
model1 = load_pickle("models/Simulation/model1_1st_order_trained.pkl")
model2 = load_pickle("models/Simulation/model2_1st_order_seasonal_trained.pkl")
model3 = load_pickle("models/Simulation/model3_1st_order_hybrid_trained.pkl")
model4 = load_pickle("models/Simulation/model4_lstm_trained.pkl")

test_data = pd.read_pickle("data/Simulation/GT_test.pkl")
T = len(set(c[1] for c in test_data.columns))

models = {"model1": model1, "model2": model2, "model3": model3, "model4": model4}
labels = {
    "model1": "Standard DBN (model 1)",
    "model2": "Seasonal DBN (model 2)",
    "model3": "Hybrid DBN (model 3)",
    "model4": "LSTM (model 4)",
}
linear = {k: v for k, v in models.items() if v.model_type == "linear"}
nonlinear = {k: v for k, v in models.items() if v.model_type != "linear"}

# =========================================================================
# 2. Full results table: KL(GT || model) over each model's own variables
#    (if a model omits GT variables, KL is against the GT marginal)
# =========================================================================
rows = []
for name, m in models.items():
    mc, mc_se = kl_monte_carlo(ground_truth, m, T, n_samples=50000)
    td, td_se = kl_on_data(ground_truth, m, test_data)
    exact_kl = (
        kl_gaussian_exact(ground_truth, m, T)
        if m.model_type == "linear" and ground_truth.model_type == "linear"
        else np.nan
    )
    rows.append({
        "model": name, "n_vars": len(m._nodes()),
        "KL_exact": exact_kl, "KL_MC": mc, "MC_se": mc_se,
        "KL_testdata": td, "testdata_se": td_se,
    })

results = pd.DataFrame(rows).set_index("model")
print("\n=== Full KL (each model on its own variables) ===")
print(results.round(4))

# =========================================================================
# 3. Common variables only: per-variable and joint KL
# =========================================================================
print(f"\n=== Common variables: {COMMON} ===")

per_var, per_var_t, joint = {}, {}, {}
for name, m in models.items():
    try:
        tbl = kl_per_variable(ground_truth, m, COMMON, T)
        per_var_t[name] = tbl
        per_var[name] = tbl.mean(axis=1)
        joint[name] = kl_joint_common(ground_truth, m, COMMON, T)
    except ValueError as e:
        print(f"{name}: skipped ({e})")

per_var_df = pd.DataFrame(per_var)
print("\nPer-variable KL, averaged over time slices:")
print(per_var_df.round(4))

joint_df = pd.Series(joint, name="KL_joint_common")
print("\nJoint KL over the common variables "
      "(exact for linear models, Gaussian-moment diagnostic for model3/model4):")
print(joint_df.round(4))

# One panel per variable, all models on each
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

for ax_, v in zip(axes.ravel(), COMMON):
    for name, tbl in per_var_t.items():
        ax_.plot(range(T), tbl.loc[v].to_numpy(dtype=float),
                 marker="o", label=labels[name])
    ax_.set_title(v)
    ax_.set_ylabel("marginal KL(GT || model)")
    ax_.set_yscale("symlog", linthresh=1e-3)   # log-like, but safe near 0

for ax_ in axes[-1]:
    ax_.set_xlabel("time slice")

axes[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# Optional: one separate figure per variable (e.g. for a report)
SAVE_SEPARATE = False
if SAVE_SEPARATE:
    for v in COMMON:
        fig, ax_ = plt.subplots(figsize=(6, 4))
        for name, tbl in per_var_t.items():
            ax_.plot(range(T), tbl.loc[v].to_numpy(dtype=float), marker="o", label=labels[name])
        ax_.set(title=v, xlabel="time slice", ylabel="marginal KL(GT || model)")
        ax_.set_yscale("symlog", linthresh=1e-3)
        ax_.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"kl_{v}.png", dpi=200)
        plt.close(fig)

# =========================================================================
# 4. Horizon sweep, all models (Monte Carlo)
# =========================================================================
horizons = [2, 3, 5, 10]
kl_by_T = {
    name: [kl_monte_carlo(ground_truth, m, h)[0] for h in horizons]
    for name, m in models.items()
}

for name, vals in kl_by_T.items():
    plt.plot(horizons, vals, marker="o", label=labels[name])
plt.xlabel("number of time slices")
plt.ylabel("KL(GT || model)")
plt.yscale("log")
plt.legend()
plt.show()

# =========================================================================
# 5. Linear models: exact sweep, gap, and all-model comparison
# =========================================================================
horizons_fine = list(range(2, 21))
exact = {n: np.array([kl_gaussian_exact(ground_truth, m, h) for h in horizons_fine])
         for n, m in linear.items()}


def kl_diff_mc(gt, a, b, T, n=50000, seed=0):
    """Paired Monte Carlo estimate of KL(a) - KL(b), valid for any pair of models."""
    x = gt.simulate(n_samples=n, n_time_slices=T, seed=seed)
    da = _gt_logp(gt, a, x, T) - a.log_likelihood(x)
    db = _gt_logp(gt, b, x, T) - b.log_likelihood(x)
    d = da - db
    return d.mean(), d.std() / np.sqrt(n)


fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

for n, v in exact.items():
    ax[0].plot(horizons_fine, v, marker="o", label=labels[n])
ax[0].set(xlabel="number of time slices", ylabel="KL(GT || model), exact",
          title="Linear DBNs (exact)")
ax[0].legend()

ax[1].plot(horizons_fine, exact["model1"] - exact["model2"], marker="o", color="k")
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set(xlabel="number of time slices", ylabel="KL(model 1) - KL(model 2)",
          title="Gap (positive = Seasonal DBN closer to GT)")

for n, v in exact.items():
    ax[2].plot(horizons_fine, v, marker="o", label=labels[n])
for n, m in nonlinear.items():
    mc = [kl_monte_carlo(ground_truth, m, h) for h in horizons]
    ax[2].errorbar(horizons, [x for x, _ in mc], yerr=[2 * s for _, s in mc],
                   marker="s", label=f"{labels[n]} (MC, ±2 se)")
ax[2].set(xlabel="number of time slices", yscale="log", title="All models (log scale)")
ax[2].legend()

plt.tight_layout()
plt.show()

print("\n=== Paired MC check of the model 1 vs model 2 gap ===")
for h in [2, 5, 10]:
    m, se = kl_diff_mc(ground_truth, models["model1"], models["model2"], h)
    ex = exact["model1"][h - 2] - exact["model2"][h - 2]
    print(f"T={h}: KL1-KL2 = {m:.4f} ± {se:.4f}  (exact: {ex:.4f})")

# =========================================================================
# 6. LSTM: spread across training seeds
# =========================================================================
seed_files = sorted(glob.glob("models/Simulation/model4_lstm_seed*.pkl"))
if seed_files:
    print("\n=== LSTM seed spread ===")
    seed_kls = []
    for path in seed_files:
        kl, se = kl_monte_carlo(ground_truth, load_pickle(path), T, n_samples=50000)
        seed_kls.append(kl)
        print(f"{path}: KL = {kl:.4f} ± {se:.4f}")
    print(f"LSTM over {len(seed_kls)} seeds: mean {np.mean(seed_kls):.4f}, "
          f"std {np.std(seed_kls, ddof=1):.4f}")

# =========================================================================
# 7. Save tables
# =========================================================================
results.to_csv("kl_full_results.csv")
per_var_df.to_csv("kl_per_variable_common.csv")
joint_df.to_csv("kl_joint_common.csv")