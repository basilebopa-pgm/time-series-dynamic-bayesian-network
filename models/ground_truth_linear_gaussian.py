from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD

# Sequence Length = 5 - 10 

# Variables
# seasonality,
#  price - log return ,
#  volume,
#  memory_index - moving avera, 
# volatility - Average true range or Bollinger Band
# trend - bull or bear on scale 0-1

model = LinearGaussianBayesianNetwork([  # Intra-time-slice t
                                       ("seasonality_0","price_0"), 
                                       ("seasonality_0","volume_0"),
                                       ("volume_0","volatility_0"),
                                       ("volatility_0", "price_0" ),
                                       ("price_0", "trend_0"),

                                        # Inter-time-slice t -> t+1
                                       ("seasonality_0","seasonality_1"), 
                                       ("volume_0","volume_1"),
                                       ("volatility_0", "volatility_1" ),
                                       ("price_0", "price_1"),
                                       ("trend_0", "trend_1"),

                                        # Intra-time-slice t+1
                                       ("seasonality_1","price_1"), 
                                       ("seasonality_1","volume_1"),
                                       ("volume_1","volatility_1"),
                                       ("volatility_1", "price_1" ),
                                       ("price_1", "trend_1"),
                                    ])




print(model.check_model())

import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph(model.edges())

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