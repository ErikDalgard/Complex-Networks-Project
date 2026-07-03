import networkx as nx
import random

def Gilbert(nodes, threshold):
    """Creates a directed garph based on Gilbert.
    Each possible directed edge i to j is included independently with probability p"""
    Gilbert = nx.Graph()
    #Adding the nodes
    for i in range(nodes):
        Gilbert.add_node(i)

    for i in range(nodes):
        for j in range(nodes):
            if i != j:
                p = random.random()
                if p < threshold:
                    Gilbert.add_edge(i, j)

    Gilbert.graph['Name']= "Gilbert"

    return Gilbert

def watts_strogatz(nodes, k, beta):
    """
    Watts-Strogatz small-world model (undirected)
    n: number of nodes
    k: each node connected to k nearest neighbors (must be even)
    beta: rewiring probability
    """
    G = nx.watts_strogatz_graph(nodes, k, beta)
    G.graph['Name'] = "Watts_strogatz"
    return G


def configuration_model(degree_sequence):
    """
    Generate a configuration model graph using NetworkX.

    degree_sequence: target degree sequence
    """

    G = nx.configuration_model(degree_sequence)

    G = nx.Graph(G)  # collapses multi-edges

    G.remove_edges_from(nx.selfloop_edges(G))

    G = nx.convert_node_labels_to_integers(G)

    G.graph['Name'] = "Configuration_model"

    return G

def barabasi_albert(n, m):
    """
    Generate a Barabási–Albert (preferential attachment) graph.
    n: number of nodes
    m: number of edges to attach from a new node to existing nodes
                (must satisfy 1 <= m < n)

    """

    G = nx.barabasi_albert_graph(n=n, m=m)

    G.graph["Name"] = "Barabasi_albert"

    return G