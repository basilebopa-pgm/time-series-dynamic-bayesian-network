import pandas as pd
import pickle
from pathlib import Path

from pgmpy_extensions.Structure_Learning import DBNStructureLearner

variables = ["US_Index", "Momentum", "Spread", "Volume", "Volatility", "Nasdaq_Return", "Random_i", "Random_ii"]
markov_order = 1

structure_learner = DBNStructureLearner(variables=variables, order=markov_order, scoring_method="bic-g", max_indegree=3, max_iter=10000, tabu_length=100, show_progress=True)

training_data = pd.read_pickle("data/Simulation/GT_train.pkl")

learned_1st_order_DAG = structure_learner.DynamicHillClimbing(training_data)
Standard_DBN_1st_order = structure_learner.build_dbn(learned_1st_order_DAG, "linear", markov_order)
Standard_DBN_1st_order.fit(training_data, estimator="mle", std_estimator="unbiased")
Standard_DBN_1st_order.check_model()


markov_order += 1

structure_learner = DBNStructureLearner(variables=variables, order=markov_order, scoring_method="bic-g", max_indegree=3, max_iter=10000, tabu_length=100, show_progress=True)

learned_2nd_order_DAG = structure_learner.DynamicHillClimbing(training_data)
Standard_DBN_2nd_order = structure_learner.build_dbn(learned_1st_order_DAG, "linear", markov_order)
Standard_DBN_2nd_order.fit(training_data, estimator="mle", std_estimator="unbiased")
Standard_DBN_2nd_order.check_model()

current_dir = Path(__file__).resolve().parent
models_dir = current_dir.parent / "pkl"
models_dir.mkdir(parents=True, exist_ok=True)

with open(models_dir / "Standard_DBN_1st_order_trained_sim.pkl","wb") as f:pickle.dump(Standard_DBN_1st_order,f)
with open(models_dir / "Standard_DBN_2nd_order_trained_sim.pkl","wb") as f:pickle.dump(Standard_DBN_2nd_order,f)

import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph(Standard_DBN_1st_order.edges())
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

plt.figure(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightgreen",
    node_size=3000,
    font_size=10,
    arrowsize=18,
    edge_color="black",
)
plt.title("Bayesian Network Hierarchical Structure")
plt.show()