import json
import networkx as nx
import matplotlib.pyplot as plt

from motif import motifCounter


fname = "sample2"

fp = open(f"./data/{fname}.json")
data = json.load(fp)

G = nx.DiGraph()

for i, node in enumerate(data["nodes"]):
    G.add_node(i, name=node["name"], isPrey=node["group"] == "Prey")


for link in data["links"]:
    G.add_edge(link["source"], link["target"])

nx.draw(G)
plt.show()


print(motifCounter(G))
