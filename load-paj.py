import networkx as nx
import matplotlib.pyplot as plt
from motif import motifCounter

fname = "Everglades"

G = nx.read_pajek(f"./paj/{fname}.paj", )
G = nx.DiGraph(G)

nx.draw(G)
plt.show()

print(motifCounter(G))
