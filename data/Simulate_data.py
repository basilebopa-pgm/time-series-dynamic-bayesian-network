import numpy as np
import pandas as pd

from models.ground_truth_model import create_ground_truth

n_samples = 1000
sequence_length = 10
seed = 42

np.random.seed(seed)

GT_model = create_ground_truth()

data = GT_model.simulate(
        n_samples=n_samples,
        n_time_slices=sequence_length,
        seed=seed,
        return_format="wide"
        )

if __name__ == "__main__":

    data.to_pickle("data/Simulation/Ground_Truth_data.pkl")
    data.to_csv("data/Simulation/Ground_Truth_data.csv",index=False)

    print("Generated data:", data.shape)
    print(data.columns.tolist())