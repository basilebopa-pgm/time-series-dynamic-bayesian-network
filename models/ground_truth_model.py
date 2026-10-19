from models.DLGBN import DynamicBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD

def create_ground_truth():
        
    ground_truth = DynamicBayesianNetwork([
        # Intra-slice
        (("Seasonality", 0), ("Log_return", 0)),
        (("Seasonality", 0), ("Volume", 0)),
        (("Volume", 0), ("Volatility", 0)),
        (("Volatility", 0), ("Log_return", 0)),
        (("Log_return", 0), ("Trend", 0)),

        # Temporal
        (("Seasonality", 0), ("Seasonality", 1)),
        (("Log_return", 0), ("Log_return", 1)),
        (("Volume", 0), ("Volume", 1)),
        (("Volatility", 0), ("Volatility", 1)),
        (("Trend", 0), ("Trend", 1)),
        (("Random", 0), ("Random", 1)),
    ])
    
    initial_cpds = [LinearGaussianCPD(("Seasonality", 0), [0.5], 0.10),
                    LinearGaussianCPD(("Volume", 0), [10.0, 0.5], 1.0, evidence=[("Seasonality", 0)]),
                    LinearGaussianCPD(("Volatility", 0), [0.2, 0.3], 0.20, evidence=[("Volume", 0)]), 
                    LinearGaussianCPD(("Log_return", 0), [50.0, 2.0, -1.5], 2.0, evidence=[("Seasonality", 0), ("Volatility", 0)]),
                    LinearGaussianCPD(("Trend", 0), [1.0, 0.05], 0.50, evidence=[("Log_return", 0)]),
                    LinearGaussianCPD(("Random", 0), [0.5], 0.10)
                    ]
    
    transition_cpds = [LinearGaussianCPD(("Seasonality", 1), [0.5, 0.8], 0.10, evidence=[("Seasonality", 0)]), 
                    LinearGaussianCPD(("Volume", 1), [10.0, 0.5, 0.7], 1.0, evidence=[("Seasonality", 1), ("Volume", 0)]), 
                    LinearGaussianCPD(("Volatility", 1), [0.2, 0.3, 0.6], 0.20, evidence=[("Volume", 1), ("Volatility", 0)]),
                    LinearGaussianCPD(("Log_return", 1), [50.0, 2.0, -1.5, 0.75], 2.0, evidence=[("Seasonality", 1), ("Volatility", 1), ("Log_return", 0)]), 
                    LinearGaussianCPD(("Trend", 1), [1.0, 0.05, 0.9], 0.50, evidence=[("Log_return", 1), ("Trend", 0)]),
                    LinearGaussianCPD(("Random", 1), [0.5, 0.8], 0.10, evidence=[("Random", 0)])
                    ]

    ground_truth.add_cpds(*initial_cpds, *transition_cpds)

    return ground_truth


# gtm = create_ground_truth()

# import matplotlib.pyplot as plt
# import networkx as nx

# G = nx.DiGraph(gtm.edges())


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
# plt.title("Ground Truth Dynamic Bayesian Network Structure")
# plt.show()