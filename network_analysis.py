#Average path lengths
#Clustering Coefficient
#Degree Distribution
import networkx as nx
import matplotlib.pyplot as plt

def APL(G):
    return nx.average_shortest_path_length(G)

def CC(G):
    return nx.average_clustering(G)

def degreeDistribution(G):
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    plt.hist(in_degrees)
    plt.title("In-degree distribution")
    plt.show()
    plt.hist(out_degrees)
    plt.title("Out-degree distribution")
    plt.show()
