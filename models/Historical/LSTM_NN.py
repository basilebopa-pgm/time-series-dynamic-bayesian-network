import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from pgmpy_extensions.LSTMModel import LSTMSequenceModel

train_data = pd.read_pickle("data/Simulation/GT_train.pkl")

variables = ["US_Index", "Momentum", "Spread", "Volume", "Volatility", "Nasdaq_Return"]

N_SEEDS = 5
runs = []

current_dir = Path(__file__).resolve().parent
models_dir = current_dir.parent / "pkl"
models_dir.mkdir(parents=True, exist_ok=True)



for seed in range(N_SEEDS):
    model = LSTMSequenceModel(variables, hidden_dim=64, num_layers=1)
    model.fit(
        train_data,
        epochs=300, lr=3e-3, batch_size=128,
        val_frac=0.15, patience=30, weight_decay=1e-5,
        seed=seed, verbose=False,
    )
    print(f"seed {seed}: best val NLL (standardised) = {model.best_val_nll:.4f}")
    runs.append(model)

    with open(models_dir / "LSTM_NN_sim_seed{seed}.pkl", "wb") as f:
        pickle.dump(model, f)

# Pick the final model by VALIDATION loss only (never by KL to the ground truth)
best = min(runs, key=lambda m: m.best_val_nll)
with open(models_dir / "LSTM_NN_sim_seed{seed}.pkl", "wb") as f:
    pickle.dump(best, f)
print("saved best model by validation NLL")