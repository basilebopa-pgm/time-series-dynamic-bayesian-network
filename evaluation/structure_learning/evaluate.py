import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from models.ground_truth_model import create_ground_truth
from evaluation.structure_learning.structure_scoring import evaluate_structure


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


ground_truth = create_ground_truth()

model1_dag = load_pickle("models/Simulation/model1_1st_order_trained.pkl")
model2_dag = load_pickle("models/Simulation/model2_1st_order_seasonal_trained.pkl")
model3_dag = load_pickle("models/Simulation/model3_1st_order_hybrid_trained.pkl")


results_model1 = evaluate_structure(ground_truth, model1_dag, "Model 1 - Random")

results_model2 = evaluate_structure(ground_truth, model2_dag, "Model 2 - Seasonality")

results_model3 = evaluate_structure(ground_truth, model3_dag, "Model 3 - Hybrid")


model_results = {
    "Model 1 - Random": results_model1,
    "Model 2 - Seasonality": results_model2,
    "Model 3 - Hybrid": results_model3,
}


metrics = ["SHD", "Precision", "Recall", "F1"]

df = pd.DataFrame({
    name: results["Combined"]
    for name, results in model_results.items()
}).T[metrics]

print(df.round(4))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, metric in zip(axes.flat, metrics):
    df[metric].plot(kind="bar", ax=ax)
    ax.set_title(metric)
    ax.set_xlabel("Model")
    ax.set_ylabel("SHD (lower is better)" if metric == "SHD"
                  else "Score (higher is better)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("DBN Structure Learning Evaluation")
fig.tight_layout()
fig.savefig("models/Simulation/structure_metrics.png", dpi=300, bbox_inches="tight")
plt.show()