# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# plot_utils.py
import matplotlib.pyplot as plt
import networkx as nx

def plot_network_topology(network, scenario_name, output_dir="ml_exports"):
    G = nx.DiGraph()

    # Add buses
    for bus in network.buses.index:
        G.add_node(bus, type="bus")

    # Add generators
    for gen, row in network.generators.iterrows():
        G.add_node(gen, type="generator")
        G.add_edge(gen, row["bus"])

    # Add lines
    for line, row in network.lines.iterrows():
        G.add_edge(row["bus0"], row["bus1"])

    # Plot layout
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    node_colors = []
    for node in G.nodes:
        if node in network.buses.index:
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgreen")

    nx.draw(G, pos, with_labels=True, node_size=800, node_color=node_colors,
            arrows=True, edge_color="gray", font_size=9)
    plt.title(f"Network Topology: {scenario_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{scenario_name}_network_topology.png")
    plt.close()

    print(f"🧭 Saved topology plot for scenario '{scenario_name}'.")
