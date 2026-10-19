import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from models.Simulation.ground_truth_model import create_ground_truth
from evaluation.parameter_estimation.kl_divergence import evaluate_parameters

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


ground_truth = create_ground_truth()
model1  = load_pickle("models/Simulation/model1_1st_order_trained.pkl")
model2  = load_pickle("models/Simulation/model2_1st_order_seasonal_trained.pkl")
model3  = load_pickle("models/Simulation/model3_1st_order_hybrid_trained.pkl")

test_data = pd.read_pickle("data/Simulation/GT_test.pkl")

results_model1 = evaluate_parameters(ground_truth, model1, test_data)
results_model2 = evaluate_parameters(ground_truth, model2, test_data)
results_model3 = evaluate_parameters(ground_truth, model3, test_data)

model_results = {
    "Model 1 - Random": results_model1,
    "Model 2 - Seasonality": results_model2,
    "Model 3 - Hybrid": results_model3,
}

df = pd.DataFrame({
    name: results["Overall"]
    for name, results in model_results.items()
}).T

print("\nOverall Parameter Evaluation")
print(df.round(4))

metrics = ["Mean Conditional KL", "RMSE"]
metrics = [metric for metric in metrics if metric in df.columns]

fig, axes = plt.subplots(
    1, len(metrics),
    figsize=(6 * len(metrics), 5),
    squeeze=False,
)

for ax, metric in zip(axes.flat, metrics):
    df[metric].plot(kind="bar", ax=ax)
    ax.set_title(metric)
    ax.set_xlabel("Model")
    ax.set_ylabel(
        "KL divergence (lower is better)"
        if "KL" in metric
        else "RMSE (lower is better)"
    )
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("DBN Parameter Learning Evaluation")
fig.tight_layout()
fig.savefig(
    "models/Simulation/parameter_metrics.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# Per-variable conditional KL
per_variable = pd.DataFrame({
    name: results["Per-variable KL"]
    for name, results in model_results.items()
}).T

print("\nPer-variable Conditional KL")
print(per_variable.round(4))


ax = per_variable.T.plot(kind="bar", figsize=(12, 6))
ax.set_title("Conditional KL Divergence by Variable")
ax.set_xlabel("Variable")
ax.set_ylabel("KL divergence (lower is better)")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "models/Simulation/parameter_kl_by_variable.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()