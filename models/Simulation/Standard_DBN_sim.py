import pandas as pd
import pickle
from pathlib import Path

from pgmpy_extensions.Structure_Learning import DBNStructureLearner

def truncate(data, variables, L):
    cols = [c for c in data.columns if c[0] in variables and c[1] < L]
    return data[cols]

variables = ["US_Index", "Momentum", "Spread", "Volume", "Volatility", "Nasdaq_Return", "Random_i", "Random_ii"]
full_train = pd.read_pickle("data/Simulation/GT_train.pkl")
SEQUENCE_LENGTHS = [10,20,30,45,60]

models_dir = Path(__file__).resolve().parent.parent / "pkl"
models_dir.mkdir(parents=True, exist_ok=True)

for L in SEQUENCE_LENGTHS:
    training_data = truncate(full_train, variables, L)

    for markov_order in [1, 2]:
        learner = DBNStructureLearner(variables=variables, order=markov_order, scoring_method="bic-g", max_indegree=3, max_iter=10000, tabu_length=100, show_progress=True)
        dag = learner.DynamicHillClimbing(training_data)
        dbn = learner.build_dbn(dag, "linear", markov_order)
        dbn.fit(training_data, estimator="mle", std_estimator="unbiased")
        dbn.check_model()

        with open(models_dir / f"Standard_DBN_{markov_order}_order_L{L}_sim.pkl", "wb") as f:
            pickle.dump(dbn, f)


# import matplotlib.pyplot as plt
# import networkx as nx

# G = nx.DiGraph(Standard_DBN_1st_order.edges())
# pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

# plt.figure(figsize=(8, 6))
# nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     node_color="lightgreen",
#     node_size=3000,
#     font_size=10,
#     arrowsize=18,
#     edge_color="black",
# )
# plt.title("Bayesian Network Hierarchical Structure")
# plt.show()