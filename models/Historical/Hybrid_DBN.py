import pandas as pd
import pickle
from pathlib import Path

from pgmpy_extensions.Structure_Learning import DBNStructureLearner

variables = ["US_Index", "Momentum", "Spread", "Volume", "Volatility", "Nasdaq_Return", "Random_i", "Random_ii"]

structure_learner = DBNStructureLearner(variables=variables,
                                        order=1,
                                        scoring_method="bic-g",
                                        max_indegree=3,
                                        max_iter=10000,
                                        tabu_length=100,
                                        show_progress=True)

training_data = pd.read_pickle("data/Simulation/GT_train.pkl")

learned_DAG = structure_learner.DynamicHillClimbing(training_data)
Hybrid_DBN = structure_learner.build_dbn(learned_DAG)
Hybrid_DBN .model_type = "neural"

model3_1st_order_hybrid.fit(    training_data,
                                hidden_dim=32,
                                epochs=500,
                                learning_rate=0.001,
                                seed=42,
                                )


current_dir = Path(__file__).resolve().parent
models_dir = current_dir.parent / "pkl"
models_dir.mkdir(parents=True, exist_ok=True)


with open(models_dir / "Hybrid_DBN_trained_sim.pkl","wb") as f:pickle.dump(model3_1st_order_hybrid,f)

import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph(model3_1st_order_hybrid.edges())

pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

plt.figure(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=3000,
    font_size=10,
    arrowsize=18,
    edge_color="black",
)
plt.title("Model 3 Hyrbid DBN ")
plt.show()
