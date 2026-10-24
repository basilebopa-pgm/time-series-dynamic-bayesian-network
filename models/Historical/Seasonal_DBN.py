import pandas as pd
import pickle
from pathlib import Path

from pgmpy_extensions.Structure_Learning import DBNStructureLearner

variables = ["US_Index", "Momentum", "Spread", "Volume", "Volatility", "Nasdaq_Return", "Cumulative_Seasonal", "HTF_Trend"]

structure_learner = DBNStructureLearner(variables=variables,
                                        order=1,
                                        scoring_method="bic-g",
                                        max_indegree=3,
                                        max_iter=10000,
                                        tabu_length=100,
                                        show_progress=True)

training_data = pd.read_pickle("data/Simulation/GT_train.pkl")

learned_DAG = structure_learner.DynamicHillClimbing(training_data)

model2_1st_order_seasonal = structure_learner.build_dbn(learned_DAG)

model2_1st_order_seasonal.fit(training_data, estimator="mle", std_estimator="unbiased")


current_dir = Path(__file__).resolve().parent
models_dir = current_dir.parent / "pkl"
models_dir.mkdir(parents=True, exist_ok=True)

with open(models_dir / "Seasonal_DBN_trained_sim.pkl","wb") as f:pickle.dump(model2_1st_order_seasonal,f)


import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph(model2_1st_order_seasonal.edges())

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
plt.title("Model 2 DBN with Seasonality")
plt.show()
