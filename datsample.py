import matplotlib.pyplot as plt
import networkx as nx

def draw_neural_network(layers):
    G = nx.DiGraph()

    pos = {}
    labels = {}
    node_colors = []

    # Assign positions and labels
    for layer_idx, layer in enumerate(layers):
        for node_idx, node in enumerate(layer):
            node_id = f"{layer_idx}-{node_idx}"
            G.add_node(node_id)
            pos[node_id] = (layer_idx, -node_idx)
            labels[node_id] = node
            if "Input" in node:
                node_colors.append('orange')
            elif "Hidden" in node:
                node_colors.append('cyan')
            elif "Output" in node:
                node_colors.append('yellow')

    # Connect nodes
    for layer_idx, layer in enumerate(layers[:-1]):
        next_layer = layers[layer_idx + 1]
        for node_idx, node in enumerate(layer):
            node_id = f"{layer_idx}-{node_idx}"
            for next_node_idx, next_node in enumerate(next_layer):
                next_node_id = f"{layer_idx + 1}-{next_node_idx}"
                G.add_edge(node_id, next_node_id)

    # Draw the network
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, node_color=node_colors, font_size=10, font_weight='bold')
    plt.title('Neural Network Diagram for Sentiment Analysis on Amazon Book Reviews')
    plt.show()

layers = [
    ["Input Layer\n(Book Reviews)"] * 5,  # Input layer with 5 neurons
    ["Hidden Layer 1\n(Feature Extraction)"] * 5,  # Hidden layer 1
    ["Hidden Layer 2\n(Feature Extraction)"] * 5,  # Hidden layer 2
    ["Hidden Layer 3\n(Feature Extraction)"] * 5,  # Hidden layer 3
    ["Output Layer\n(Sentiment Classification)"] * 3  # Output layer with 3 neurons
]

draw_neural_network(layers)
