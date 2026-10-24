import pickle
from pathlib import Path
import numpy as np
import pandas as pd

n_samples, sequence_length, seed = 1000, 60, 42
P, BURN = 5, 10

with open(Path("models/pkl") / "gtm_2nd_order.pkl", "rb") as f:
    gtm = pickle.load(f)

np.random.seed(seed)
wide = gtm.simulate(n_samples=n_samples, n_time_slices=sequence_length,
                    seed=seed, return_format="wide")

# wide -> long 
wide = wide.reset_index(drop=True)
df = pd.concat([
    pd.DataFrame({"sample": np.arange(len(wide)),
                  "var": c[0], "t": int(c[1]), "value": wide[c].values})
    for c in wide.columns
])
df = df.pivot_table(index=["sample", "t"], columns="var", values="value").reset_index()
df = df.sort_values(["sample", "t"]).reset_index(drop=True)

# HTF_Trend
logp = df.groupby("sample")["Nasdaq_Return"].cumsum()
df["HTF_Trend"] = logp - logp.groupby(df["sample"]).transform(lambda x: x.ewm(span=20, adjust=False).mean())

# Seasonal_index (profile fitted on first 80% of sequences)
train = df["sample"] < int(0.8 * n_samples)
z = lambda s: (s - s[train].mean()) / s[train].std()
comp = (z(df["Volume"]) + z(df["Volatility"]) + z(df["Nasdaq_Return"].abs())) / 3
phase = df["t"] % P
profile = comp[train].groupby(phase[train]).mean()
df["Seasonal_index"] = (profile - profile.mean()).reindex(range(P)).values[phase]

# random controls
rng = np.random.default_rng(seed)
df["Random_i"], df["Random_ii"] = rng.standard_normal((2, len(df)))

# drop warm-up
df = df[df["t"] >= BURN].reset_index(drop=True)

# one dataset: sample, t, base, seasonal/HTF features, random controls, hidden drivers
cols = ["sample", "t",
        "US_Index", "Momentum", "Spread", "Volume", "Volatility", "Nasdaq_Return",
        "Random_i", "Random_ii", "Seasonal_index", "HTF_Trend",
        "Seasonality", "Seasonality_Q"]
df = df[cols].copy()
df["t"] = df["t"] - df["t"].min()                       # restart time at 0 after burn-in

variables = [c for c in cols if c not in ("sample", "t")]
out = df.pivot(index="sample", columns="t", values=variables)
out.columns = [(v, int(t)) for v, t in out.columns]     # (variable, time_slice) tuples
out = out.reset_index(drop=True)

out.to_pickle("data/Simulation/GT_data.pkl")
print(out.shape)                                        # (1000, 12 * 50)