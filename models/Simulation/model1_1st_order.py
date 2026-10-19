import pandas as pd
import pickle

from pgmpy_extensions.Structure_Learning import DBNStructureLearner

variables = ["Random", "Volume", "Volatility", "Log_return", "Trend"]

structure_learner = DBNStructureLearner(variables=variables,
                                        markov_order=1,
                                        scoring_method="bic-g",
                                        max_indegree=3,
                                        max_iter=10000,
                                        tabu_length=100,
                                        show_progress=True)

training_data = pd.read_pickle("data/Simulation/GT_train.pkl")

learned_DAG = structure_learner.DynamicHillClimbing(training_data)

model1_1st_order = structure_learner.build_dbn(learned_DAG)

model1_1st_order.fit(
        training_data,
        estimator="mle",
        std_estimator="unbiased"
        )

with open("models/Simulation/model1_1st_order_DAG.pkl","wb") as f:pickle.dump(learned_DAG, f)

with open("models/Simulation/model1_1st_order_trained.pkl","wb") as f:pickle.dump(model1_1st_order,f)










print()
print("===================================")
print("LEARNED INTRA-SLICE EDGES")
print("===================================")

for edge in learned_DAG["intra_edges"]:
    print(
        f"{edge[0]} -> {edge[1]}"
    )


print()
print("===================================")
print("LEARNED INTER-SLICE EDGES")
print("===================================")

for edge in learned_DAG["inter_edges"]:
    print(
        f"{edge[0]}[t-1] -> {edge[1]}[t]"
    )



import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph(model1_1st_order.edges())

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