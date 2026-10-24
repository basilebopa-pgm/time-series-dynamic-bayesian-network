import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from pgmpy_extensions.LSTMModel import LSTMSequenceModel

def truncate(data, variables, L):
    cols = [c for c in data.columns if c[0] in variables and c[1] < L]
    return data[cols]


train_data = pd.read_pickle("data/Simulation/GT_train.pkl")
SEQUENCE_LENGTHS = [10,20,30,45,60]
variables = ["US_Index", "Momentum", "Spread", "Volume", "Volatility", "Nasdaq_Return"]
N_SEEDS = 5
models_dir = Path(__file__).resolve().parent.parent / "pkl"
models_dir.mkdir(parents=True, exist_ok=True)

for L in SEQUENCE_LENGTHS:
    data_L = truncate(train_data, variables, L)
    runs = []

    for seed in range(N_SEEDS):
        model = LSTMSequenceModel(variables, hidden_dim=64, num_layers=1)
        model.fit(data_L, epochs=300, lr=3e-3, batch_size=128,
                  val_frac=0.15, patience=30, weight_decay=1e-5,
                  seed=seed, verbose=False)
        print(f"L={L} seed {seed}: best val NLL = {model.best_val_nll:.4f}")
        runs.append(model)

        with open(models_dir / f"LSTM_NN_sim_L{L}_seed{seed}.pkl", "wb") as f:
            pickle.dump(model, f)

    # pick by VALIDATION loss only, never by KL to the ground truth
    best = min(runs, key=lambda m: m.best_val_nll)
    with open(models_dir / f"LSTM_NN_sim_L{L}_best.pkl", "wb") as f:
        pickle.dump(best, f)
    print(f"L={L}: saved best by validation NLL")