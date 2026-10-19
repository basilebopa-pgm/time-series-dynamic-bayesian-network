from models.DLGBN import DynamicBayesianNetwork

model_1 = DynamicBayesianNetwork([
    # Intra-slice
    (("Volatility", 0), ("Price", 0)),
    (("Price", 0), ("Trend", 0)),

    # Temporal
    (("Price", 0), ("Price", 1)),
    (("Volume", 0), ("Volume", 1)),
    (("Volatility", 0), ("Volatility", 1)),
    (("Trend", 0), ("Trend", 1)),
])

dbn = DynamicBayesianNetwork(markov_order=3)

dbn.add_edge(("Price", 0), ("Price", 1))
dbn.add_edge(("Price", 0), ("Price", 2))
dbn.add_edge(("Price", 0), ("Price", 3))
dbn.add_edge(("Price", 1), ("Price", 2))
dbn.add_edge(("Price", 1), ("Price", 3))
dbn.add_edge(("Price", 2), ("Price", 3))


import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph(model_1.edges())
G2 = nx.DiGraph(dbn.edges())
# Uses Graphviz 'dot' layout to organize nodes by hierarchy (top-to-bottom)
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
pos2 = nx.nx_agraph.graphviz_layout(G2, prog="dot")

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
plt.title("Bayesian Network Hierarchical Structure")
plt.show()