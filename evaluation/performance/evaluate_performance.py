import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.Simulation.ground_truth_model import create_ground_truth
from evaluation.performance.performance import (
    forecast_metrics, persistence_metrics, cond_log_score, COMMON_VARS
)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================================
# 1. Load
# =========================================================================
ground_truth = create_ground_truth()
models = {
    "GT": ground_truth,   # oracle reference: the best any model can do
    "model1 1st_order": load_pickle("models/Simulation/model1_1st_order_trained.pkl"),
    "model 2nd order": load_pickle("models/Simulation/model1_2nd_order_trained.pkl"),
    "model2": load_pickle("models/Simulation/model2_1st_order_seasonal_trained.pkl"),
    "model3": load_pickle("models/Simulation/model3_1st_order_hybrid_trained.pkl"),
    "model4": load_pickle("models/Simulation/model4_lstm_trained.pkl"),
}
labels = {
    "GT": "Ground truth (oracle)",
    "model1 1st_order": "Standard DBN (model 1 markov = 1)",
    "model1 2nd order": "Standard DBN (model 1 markov = 2)",
    "model2": "Seasonal DBN (model 2)",
    "model3": "Hybrid DBN (model 3)",
    "model4": "LSTM (model 4)",
    "persistence": "Persistence (naive)",
}

test_data = pd.read_pickle("data/Simulation/GT_test.pkl")
T = len(set(c[1] for c in test_data.columns))
CONTEXT = min(3, T - 1)       # slices observed before a multi-step forecast
N_SAMPLES = 200
VARS = COMMON_VARS
print(f"{len(test_data)} test rows, {T} time slices, multi-step context = {CONTEXT}")

# =========================================================================
# 2. One-step-ahead: predict slice k from slices 0..k-1, averaged over k
# =========================================================================
one_step = []
for name, m in models.items():
    for k in range(1, T):
        df = forecast_metrics(m, test_data, context=k, variables=VARS,
                              n_steps=1, n_samples=N_SAMPLES, seed=0)
        df["model"] = name
        one_step.append(df)
for k in range(1, T):
    df = persistence_metrics(test_data, context=k, variables=VARS, n_steps=1)
    df["model"] = "persistence"
    one_step.append(df)
one_step = pd.concat(one_step, ignore_index=True)

summary = (one_step.groupby(["model", "variable"])[["rmse", "mae", "crps", "cov90", "width90"]]
           .mean().reset_index())

print("\n=== One-step-ahead forecast (mean over all context lengths) ===")
for metric in ["rmse", "crps", "cov90"]:
    tbl = summary.pivot(index="variable", columns="model", values=metric)
    cols = [c for c in ["GT", "model1", "model2", "model3", "model4", "persistence"] if c in tbl]
    print(f"\n{metric.upper()}")
    print(tbl[cols].round(4))

# Error relative to the oracle: 1.00 = as good as the true model
rel = summary.pivot(index="variable", columns="model", values="rmse")
rel = rel.div(rel["GT"], axis=0)
print("\nRMSE relative to ground truth (1.00 = oracle):")
print(rel[[c for c in ["model1", "model2", "model3", "model4", "persistence"] if c in rel]].round(3))

# =========================================================================
# 3. Multi-step: observe CONTEXT slices, forecast the rest
# =========================================================================
multi = []
for name, m in models.items():
    df = forecast_metrics(m, test_data, context=CONTEXT, variables=VARS,
                          n_samples=N_SAMPLES, seed=0)
    df["model"] = name
    multi.append(df)
df = persistence_metrics(test_data, context=CONTEXT, variables=VARS)
df["model"] = "persistence"
multi.append(df)
multi = pd.concat(multi, ignore_index=True)

order = [m for m in ["GT", "model1", "model2", "model3", "model4", "persistence"]
         if m in multi["model"].unique()]


def grid_plot(metric, ylabel, fname, ref=None, log=False):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax_, v in zip(axes.ravel(), VARS):
        for name in order:
            d = multi[(multi.model == name) & (multi.variable == v)].sort_values("horizon")
            if d[metric].isna().all():
                continue
            style = "--" if name in ("GT", "persistence") else "-"
            ax_.plot(d["horizon"], d[metric], style, marker="o", label=labels[name])
        if ref is not None:
            ax_.axhline(ref, color="gray", lw=0.8)
        if log:
            ax_.set_yscale("log")
        ax_.set_title(v)
        ax_.set_ylabel(ylabel)
    for ax_ in axes[-1]:
        ax_.set_xlabel("forecast horizon (slices ahead)")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.show()


grid_plot("rmse", "RMSE of predictive mean", "perf_rmse_by_horizon.png", log=True)
grid_plot("crps", "CRPS (lower is better)", "perf_crps_by_horizon.png", log=True)

# Calibration: central 90% intervals should cover about 90% of observations
cov = multi[multi.model != "persistence"].groupby(["model", "horizon"])["cov90"].mean().unstack(0)
cov = cov[[c for c in order if c in cov]]
for name in cov:
    plt.plot(cov.index, cov[name], marker="o", label=labels[name])
plt.axhline(0.9, color="gray", lw=0.8)
plt.xlabel("forecast horizon (slices ahead)")
plt.ylabel("coverage of central 90% interval")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("perf_coverage_by_horizon.png", dpi=200)
plt.show()

print("\n=== Multi-step forecast: averaged over horizons ===")
ms = multi.groupby(["model", "variable"])[["rmse", "crps", "cov90", "width90"]].mean().reset_index()
for metric in ["rmse", "crps", "cov90"]:
    tbl = ms.pivot(index="variable", columns="model", values=metric)
    print(f"\n{metric.upper()}")
    print(tbl[[c for c in order if c in tbl]].round(4))

# =========================================================================
# 4. Predictive log score (comparable only between models with the same variables)
# =========================================================================
print("\n=== Predictive log score of the future given the first "
      f"{CONTEXT} slices (higher is better) ===")
rows = []
for name, m in models.items():
    rows.append({"model": name, "n_vars": len(m._nodes()),
                 "log_score_per_slice": cond_log_score(m, test_data, CONTEXT)})
log_score = pd.DataFrame(rows).set_index("model")
print(log_score.round(3))

# =========================================================================
# 5. Save
# =========================================================================
summary.to_csv("perf_one_step.csv", index=False)
multi.to_csv("perf_multi_step.csv", index=False)
log_score.to_csv("perf_log_score.csv")