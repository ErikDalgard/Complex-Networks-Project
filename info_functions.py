import os
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import math

def general_info(G):
    """
    Given a networkx graph G returns how many nodes, edges"""
    print(f"General info for network: {G.graph['Name']}")
    print("----------------------------------")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f"Network has {n_nodes} nodes and {n_edges} edges.")
    # Connected components (undirected analogue)
    cc = max(nx.connected_components(G), key=len)
    G_cc = G.subgraph(cc)

    print(f"Largest connected component: {G_cc.number_of_nodes()} nodes, {G_cc.number_of_edges()} edges")
    print(f"CC is {G_cc.number_of_nodes()/n_nodes*100} % of nodes, {G_cc.number_of_edges()/n_edges*100} % of edges ")
    print("----------------------------------\n")



def distance_info(G, sample_size=1000, save_img=False):
    """Given a networkx graph G returns the average shortest distance in the CC. 
    Uses sampling to speed up computation."""
    
    print(f"Distance info for network: {G.graph['Name']}")
    print("----------------------------------")
    
    cc = max(nx.connected_components(G), key=len)
    G_cc = G.subgraph(cc)
    n = len(G_cc)

    # Sample nodes if graph is large
    nodes = list(G_cc.nodes())
    sample = random.sample(nodes, min(sample_size, n))

    all_lengths = []
    for source in sample:
        lengths = nx.single_source_shortest_path_length(G_cc, source)
        all_lengths.extend(l for t, l in lengths.items() if t != source)

    avg_distance = sum(all_lengths) / len(all_lengths)
    length_counter = Counter(all_lengths)

    print(f"The average shortest distance (sampled from {len(sample)} nodes) is {avg_distance:.2f}.")

    #Picking a scale for the y-axis so small graphs dont show up as all zeros
    max_count = max(length_counter.values())
    if max_count >= 1e6:
        scale, scale_label = 1e6, "×10⁶"
    elif max_count >= 1e3:
        scale, scale_label = 1e3, "×10³"
    else:
        scale, scale_label = 1, ""

    plt.figure(figsize=(8, 5))
    plt.bar(length_counter.keys(), length_counter.values())
    plt.xlabel("Shortest Path Length")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/scale:.1f}'))
    plt.ylabel(f"Number of Pairs ({scale_label})" if scale_label else "Number of Pairs")
    plt.title(f"Distance Distribution of network {G.graph['Name']}")
    if save_img:
        os.makedirs("Images/Distance", exist_ok=True)
        plt.savefig(f"Images/Distance/{G.graph['Name']}")
    plt.show()
    print("----------------------------------", end="\n\n")

def degree_info(G, save_img=False):
    """Given a networkx graph G, returns the largest in, out and total degree. Also plots the degree distrubution of in, out and total.Fits a power law and normal
    """
    print(f"Degree info for network: {G.graph['Name']}")
    print("----------------------------------")
    G_total_degrees = np.array([d for n, d in G.degree()])
    total_bin_counts = np.bincount(G_total_degrees)

    degrees_lst = range(len(total_bin_counts))

    mask = (total_bin_counts > 0) & (np.arange(len(degrees_lst)) > 0)
    x = np.arange(len(degrees_lst))[mask]
    y = total_bin_counts[mask]

    coeffs = np.polyfit(np.log(x), np.log(y), 1)
    alpha, beta = -coeffs[0], coeffs[1]

    y_fit = np.exp(beta) * x**(-alpha)

    print(f"The largest degree of a node is {np.max(G_total_degrees)}")
    print(f"The mean degree of all nodes is {np.mean(G_total_degrees)}")
    print(f"The variance of the degrees are: {np.sum((G_total_degrees-np.mean(G_total_degrees))**2) / len(G_total_degrees)}")
    #Calculating the degree distrubution of in, out and total degrees. THen counting the bins and plotting it in a loglog.
    plt.loglog(degrees_lst, total_bin_counts, 'o', label='degree')
    plt.loglog(x, y_fit, '--', label=f"Power law fit")

    plt.plot()
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.legend()
    if save_img:
        os.makedirs("Images/Degree", exist_ok=True)
        plt.savefig(f"Images/Degree/{G.graph['Name']}_loglog")
    plt.show()


    #Degree distribution - fraction of nodes.
    G_degrees_dict = dict(G.degree())
    degrees = list(G_degrees_dict.values())
    mean_degree = np.mean(degrees)
    std_degree = np.std(degrees)

    def gaussian(x, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    x = np.linspace(mean_degree - 4*std_degree, mean_degree + 4*std_degree, 1000)

    x_range = np.arange(x.min(), max(G_degrees_dict.values()) + 1)
    y_fit = np.exp(beta) * x_range ** (-alpha)

    plt.hist(list(G_degrees_dict.values()), bins=range(0, max(G_degrees_dict.values()) + 1), density=True, label="Degree distrubution")
    plt.plot(x, gaussian(x, mean_degree, std_degree), label="Gaussian Fit", alpha=0.6)
    plt.plot(x_range, y_fit/G.number_of_nodes(), 'r--', label=f"Power law fit", alpha=0.6)
    plt.xlabel("Degree")
    plt.ylabel("Fraction of Nodes")
    plt.title("Degree Distribution with Gaussian Fit")
    plt.ylim(0, 0.5)
    plt.legend()
    if save_img:
        os.makedirs("Images/Degree", exist_ok=True)
        plt.savefig(f"Images/Degree/{G.graph['Name']}")
    plt.show()
    print("----------------------------------", end="\n\n")


def clustering_info(G):
    clust = nx.average_clustering(G)
    print(f"Clustering coefficient on network {G.graph['Name']}: {100 * clust:.4f}%")
    print("----------------------------------")



def hetoero_info(G):
    print(f"Hetero info for network: {G.graph['Name']}")
    print("----------------------------------")
    
    degrees = np.array([d for _, d in G.degree()])


    k_mean = degrees.mean()
    k2_mean = (degrees**2).mean()

    H = k2_mean / (k_mean**2)

    print("Heterogeneity:", H)
    print("----------------------------------", end="\n\n")



def assortativity_info(graphs, save_img=False):
    """
    Given a list of NetworkX graphs, prints the degree assortativity
    coefficient for each graph and plots their degree correlation
    functions in a single figure.
    """

    n_graphs = len(graphs)

    #Subplots
    ncols = 3
    nrows = math.ceil(n_graphs / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharex=True,
        sharey=True
    )

    axes = np.array(axes).flatten()

    for ax, G in zip(axes, graphs):

        # Degree assortativity
        deg_corr = nx.degree_assortativity_coefficient(G)
        print(f"Degree correlation is: {deg_corr} for network {G.graph['Name']}")

        # Degree correlation function
        knn = nx.average_degree_connectivity(G)
        degrees = np.array(sorted(knn.keys()))
        avg_neighbor_degree = np.array([knn[d] for d in degrees])

        ax.scatter(degrees, avg_neighbor_degree)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(G.graph["Name"])
        ax.set_xlabel("Degree $k$")
        ax.set_ylabel(r"$k_{nn}(k)$")

        ax.set_xlim(0.8, 300)
        ax.set_ylim(4, 300)

    # Remove any unused subplots
    for ax in axes[n_graphs:]:
        fig.delaxes(ax)

    plt.tight_layout()

    if save_img:
        os.makedirs("Images/Correlation", exist_ok=True)
        plt.savefig("Images/Correlation/DegreeCorrelationComparison.png",
                    dpi=300, bbox_inches="tight")

    plt.show()



def robust_info(G, save_img=False):
    """
    Creates a targeted and random attack on a network to check how robust it is. Plots the results.
    
    """

    core_copy_random = G.copy()
    core_copy_targeted = G.copy()
    num_steps = 25
    M = G.number_of_nodes() // num_steps
    num_nodes_removes = range(0, G.number_of_nodes(), M)

    attack_porportion_random = []
    attack_porportion_targeted = []
    for nodes_remove in num_nodes_removes:
        #RANDOM ATTACK
        giant_random = max(nx.connected_components(core_copy_random), key=len)
        core_proportion_random = len(giant_random) / G.number_of_nodes()
        attack_porportion_random.append(core_proportion_random)

        #Sampling M nodes at random and removing
        nodes_to_remove_random = random.sample(list(giant_random), min(M, len(giant_random)))
        core_copy_random.remove_nodes_from(nodes_to_remove_random)


        #TARGETED ATTACK
        #getting the new giant component
        giant_targeted = max(nx.connected_components(core_copy_targeted), key=len)
        subgraph_nodes = list(giant_targeted)
        core_proportion_targeted = len(giant_targeted) / G.number_of_nodes()
        attack_porportion_targeted.append(core_proportion_targeted)

        #Choosing the M biggest hubs to remove
        sorted_nodes = sorted(subgraph_nodes, key=core_copy_targeted.degree, reverse=True)
        nodes_to_remove_targeted = sorted_nodes[:M]

        #Removing them
        core_copy_targeted.remove_nodes_from(nodes_to_remove_targeted)
    
    

    plt.title(f"Attack on network: {G.graph['Name']}")
    plt.plot(num_nodes_removes, attack_porportion_random, 'o-', label="Random")
    plt.plot(num_nodes_removes, attack_porportion_targeted, 'o-', label="Targeted")
    plt.xlabel("Nodes removed")
    plt.ylabel("Proportion in giant component")
    plt.legend()
    if save_img:
        os.makedirs("Images/Robust", exist_ok=True)
        plt.savefig(f"Images/Robust/{G.graph['Name']}")
    plt.show()


def community_info(G, save_img=False):
    """
    Given a networkx graph G, returns the number of communities and plots the communities.
    """ 

    print("COMMUNITY INFO ABOUT NETWORK:", G.graph['Name'], end="\n\n")

    #Number of communities
    partition = nx.community.greedy_modularity_communities(G)
    print(f"Number of communities: {len(partition)}")

    #Modularity score
    modularity_score = nx.community.modularity(G, partition)
    print(f"Modularity score: {modularity_score:.4f}")

    #Community size
    sizes = sorted([len(c) for c in partition], reverse=True)
    avg_size = np.mean(sizes)
    largest = sizes[0]
    smalles = sizes[-1]

    print(f"Average community size: {avg_size:.2f}")
    print(f"Largest community size: {largest}") 
    print(f"Smallest community size: {smalles}")

    print("----------------------------------", end="\n\n")
    

    plt.hist(sizes, bins=30)
    plt.xlabel("Community Size")
    plt.ylabel("Count")
    plt.title(f"Community Size Distribution for {G.graph['Name']}")
    if save_img:
        os.makedirs("Images/Communities", exist_ok=True)
        plt.savefig(f"Images/Communities/{G.graph['Name']}")
    plt.show()
    return partition

def homophily_info(G, partion):
    node_to_community = {}
    for i, community in enumerate(partion):
        for node in community:
            node_to_community[node] = i

    #Count cross-community edges
    total_edges = G.number_of_edges()
    cross_edges = sum(1 for u, v in G.edges() 
                    if node_to_community[u] != node_to_community[v])

    fraction_cross = cross_edges / total_edges

    #Expected cross edges under random mixing
    n = G.number_of_nodes()
    community_sizes = Counter(node_to_community.values())
    expected_cross = 1 - sum((size/n)**2 for size in community_sizes.values())

    #Results
    print(f"Total edges: {total_edges}")
    print(f"Cross-community edges: {cross_edges}")
    print(f"Fraction cross-community: {fraction_cross:.4f}")
    print(f"Expected cross (random 2pq): {expected_cross:.4f}")
    print(f"Homophily signal: {fraction_cross < expected_cross}")
    print("----------------------------------", end="\n\n")

def closeness_info(G, save_img=False):
    "Given nx graph G plot the closeness distribution"

    print(f"Closeness info for network: {G.graph['Name']}")
    print("----------------------------------")
    closeness = nx.closeness_centrality(G)
    print(f"Mean closeness: {np.mean(list(closeness.values()))}")
    print(f"Max closeness : {np.max(list(closeness.values()))}")

    #Plotting distrubution of closeness
    plt.title("Distrubution of closeness for network: " + G.graph['Name'])
    plt.ylabel("Count")
    plt.xlabel("Closeness")
    plt.hist(list(closeness.values()))

    if save_img:
        os.makedirs("Images/Closeness", exist_ok=True)
        plt.savefig(f"Images/Closeness/{G.graph['Name']}")
    plt.show()

    

def betweeness_info(G, save_img=False):
    "Given nx graph G plot the betweeness distribution"

    print(f"Betweeness info for network: {G.graph['Name']}")
    print("----------------------------------")

    #plotting distribution of betweeness
    betweeness = list(nx.betweenness_centrality(G).values())
    print(f"Mean betweeness: {np.mean(list(betweeness))}")
    print(f"Max betweeness : {np.max(list(betweeness))}")

    plt.hist(betweeness)
    plt.title("Distrubution of betweeness for network: " + G.graph['Name'])
    plt.ylabel("Count")
    plt.xlabel("Betweeness")
    if save_img:
        os.makedirs("Images/Betweeness", exist_ok=True)
        plt.savefig(f"Images/Betweeness/{G.graph['Name']}")
    plt.show()