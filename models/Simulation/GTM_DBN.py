import pickle
from pathlib import Path
import numpy as np
from pgmpy_extensions.DLGBN import DynamicBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD

P     = 5                      # period in time steps
omega = 2 * np.pi / P
rho   = 0.99                    # damping: 1.0 = pure sinusoid, lower = more stochastic
sigma = np.sqrt(1 - rho**2)     # keeps each component's stationary variance at 1
c, s  = rho * np.cos(omega), rho * np.sin(omega)
phi0 = 0.0 

# Intra-slice 
intra_slice_edges = [
                (("Seasonality", 0), ("Volatility",0)),
                (("Seasonality", 0),("Volume", 0)),   
                (("US_Index", 0),("Volatility",0)),
                (("US_Index",0),("Nasdaq_Return",0)),
                (("Momentum",0),("Nasdaq_Return",0)),
                (("Spread",0),("Nasdaq_Return", 0)),
                (("Volatility",0),("Spread",0)),
                (("Volume", 0),("Spread", 0)),
]


# Lag 1 (t-1 -> t)
lag1_edges = [
                (("Seasonality",   0), ("Seasonality",   1)),
                (("Seasonality_Q", 0), ("Seasonality",   1)),
                (("Seasonality",   0), ("Seasonality_Q", 1)),
                (("Seasonality_Q", 0), ("Seasonality_Q", 1)),
                (("US_Index",0),("US_Index",1)),
                (("Momentum", 0),("Momentum", 1)),
                (("Volatility",0),("Volatility", 1)),
                (("Volume",0),("Volume",1)),
]

# Lag 2 (t-2 -> t)
lag2_edges = [
                (("US_Index",0),("US_Index",2)),
                (("Volume",0),("Volume",2)),
                (("Volatility",0),("Volatility",2)),
                (("Nasdaq_Return", 0),("Momentum", 2)),
]

# ---- Slice 0: initial (approx. stationary marginals) ----
slice0 = [
        LinearGaussianCPD(("Seasonality",   0), [np.cos(phi0)], 0.1),
        LinearGaussianCPD(("Seasonality_Q", 0), [np.sin(phi0)], 0.1),

        LinearGaussianCPD(("US_Index", 0),      [0.0], 1.0),
        LinearGaussianCPD(("Momentum", 0),      [0.0], 1.0),
        LinearGaussianCPD(("Spread", 0),        [0.0, 0.5, -0.3], 0.80,                   evidence=[("Volatility", 0), ("Volume", 0)]),
        LinearGaussianCPD(("Volatility", 0),    [0.0, 0.2, -0.15], 0.97,                  evidence=[("Seasonality", 0), ("US_Index", 0)]),
        LinearGaussianCPD(("Volume", 0),        [0.0, 0.3], 0.95,                         evidence=[("Seasonality", 0)]),
        LinearGaussianCPD(("Nasdaq_Return", 0), [0.0004, 0.004, 0.002, -0.002], 0.010,    evidence=[("US_Index", 0), ("Momentum", 0), ("Spread", 0)]),
        ]

# ---- Slice 1: intra + lag-1 parents ----
slice1 = [
        LinearGaussianCPD(("Seasonality",   1), [0.0,  c, -s], sigma,                     evidence=[("Seasonality", 0), ("Seasonality_Q", 0)]),
        LinearGaussianCPD(("Seasonality_Q", 1), [0.0,  s,  c], sigma,                     evidence=[("Seasonality", 0), ("Seasonality_Q", 0)]),
        LinearGaussianCPD(("US_Index", 1),      [0.0, 0.625], 0.78,                       evidence=[("US_Index", 0)]),
        LinearGaussianCPD(("Momentum", 1),      [0.0, 0.7], 0.71,                         evidence=[("Momentum", 0)]),
        LinearGaussianCPD(("Spread", 1),        [0.0, 0.5, -0.3], 0.80,                   evidence=[("Volatility", 1), ("Volume", 1)]),
        LinearGaussianCPD(("Volatility", 1),    [0.0, 0.2, -0.15, 0.625], 0.74,           evidence=[("Seasonality", 1), ("US_Index", 1), ("Volatility", 0)]),
        LinearGaussianCPD(("Volume", 1),        [0.0, 0.3, 0.47], 0.83,                   evidence=[("Seasonality", 1), ("Volume", 0)]),
        LinearGaussianCPD(("Nasdaq_Return", 1), [0.0004, 0.004, 0.002, -0.002], 0.010,    evidence=[("US_Index", 1), ("Momentum", 1), ("Spread", 1)]),
]

# ---- Slice 2: repeating template (intra + lag-1 + lag-2) ----
# beta = [intercept, *coefficients in evidence order]
template = [
        LinearGaussianCPD(("Seasonality",   2), [0.0,  c, -s], sigma,                     evidence=[("Seasonality", 1), ("Seasonality_Q", 1)]),
        LinearGaussianCPD(("Seasonality_Q", 2), [0.0,  s,  c], sigma,                     evidence=[("Seasonality", 1), ("Seasonality_Q", 1)]),
        LinearGaussianCPD(("US_Index", 2),      [0.0, 0.5, 0.2], 0.765,                   evidence=[("US_Index", 1), ("US_Index", 0)]),
        LinearGaussianCPD(("Momentum", 2),      [0.0, 0.7, 15.0], 0.70,                   evidence=[("Momentum", 1), ("Nasdaq_Return", 0)]),
        LinearGaussianCPD(("Volume", 2),        [0.0, 0.3, 0.4, 0.15], 0.80,              evidence=[("Seasonality", 2), ("Volume", 1), ("Volume", 0)]),
        LinearGaussianCPD(("Volatility", 2),    [0.0, 0.2, -0.15, 0.5, 0.2], 0.72,        evidence=[("Seasonality", 2), ("US_Index", 2),("Volatility", 1), ("Volatility", 0)]),
        LinearGaussianCPD(("Spread", 2),        [0.0, 0.5, -0.3], 0.80,                   evidence=[("Volatility", 2), ("Volume", 2)]),
        LinearGaussianCPD(("Nasdaq_Return", 2), [0.0004, 0.004, 0.002, -0.002], 0.010,    evidence=[("US_Index", 2), ("Momentum", 2), ("Spread", 2)]),
]

gtm_1st_order = DynamicBayesianNetwork(intra_slice_edges + lag1_edges, model_type="linear", order=1)
gtm_1st_order.add_cpds(*slice0, *slice1)
gtm_1st_order.check_model()

gtm_2nd_order = DynamicBayesianNetwork(intra_slice_edges + lag1_edges + lag2_edges, model_type="linear", order=2)
gtm_2nd_order.add_cpds(*slice0, *slice1, *template)
gtm_2nd_order.check_model()

current_dir = Path(__file__).resolve().parent

models_dir = current_dir.parent / "pkl"
models_dir.mkdir(parents=True, exist_ok=True)

with open(models_dir / "gtm_1st_order.pkl", "wb") as f:
    pickle.dump(gtm_1st_order, f)

with open(models_dir / "gtm_2nd_order.pkl", "wb") as f:
    pickle.dump(gtm_2nd_order, f)

import matplotlib.pyplot as plt
import networkx as nx

G1 = nx.DiGraph(gtm_1st_order.edges())
G2 = nx.DiGraph(gtm_2nd_order.edges())

pos1 = nx.nx_agraph.graphviz_layout(G1, prog="dot")
pos2 = nx.nx_agraph.graphviz_layout(G2, prog="dot")

plt.figure(figsize=(8, 6))
nx.draw(
     G1,
     pos1,
     with_labels=True,
     node_color="lightgreen",
     node_size=3000,
     font_size=10,
     arrowsize=18,
     edge_color="black",
)
plt.title("Ground Truth Dynamic Bayesian Network Structure (markov order = 1)")
plt.show()           

plt.figure(figsize=(8, 6))
nx.draw(
     G2,
     pos2,
     with_labels=True,
     node_color="lightgreen",
     node_size=3000,
     font_size=10,
     arrowsize=18,
     edge_color="black",
)
plt.title("Ground Truth Dynamic Bayesian Network Structure (markov order = 2)")
plt.show()           