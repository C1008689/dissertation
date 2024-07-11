import matplotlib.pyplot as plt
import numpy as np

def plot_neural_network_contextual():
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define the number of neurons in each layer
    input_layer = 5
    hidden_layers = [7, 7, 5]
    output_layer = 3

    # Colors for layers
    colors = ['#FFA07A', '#20B2AA', '#9370DB', '#FFD700']

    # Positions of neurons
    def get_positions(n, layer, total_layers):
        x = layer * 1.5  # Increase horizontal spacing
        y_start = (total_layers - n) / 2
        return [(x, y_start + i) for i in range(n)]

    # Plot neurons
    layers = [input_layer] + hidden_layers + [output_layer]
    total_layers = len(layers)
    positions = {}
    for i, layer in enumerate(layers):
        pos = get_positions(layer, i, total_layers)
        positions[i] = pos
        for (x, y) in pos:
            circle = plt.Circle((x, y), radius=0.15, edgecolor='k', facecolor=colors[i % len(colors)], zorder=3)
            ax.add_patch(circle)
            # Add node annotations
            plt.text(x, y, f'{i+1}-{y:.1f}', fontsize=8, ha='center', va='center', zorder=5)

    # Plot connections with different styles for better visibility
    for i in range(len(layers) - 1):
        for (x1, y1) in positions[i]:
            for (x2, y2) in positions[i + 1]:
                line = plt.Line2D([x1, x2], [y1, y2], color='k', alpha=0.5, linestyle='--' if i % 2 == 0 else '-', linewidth=0.7)
                ax.add_line(line)

    # Annotations
    plt.text(-0.75, input_layer / 2 - 0.5, 'Input Layer\n(Book Reviews)', fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    for i in range(1, len(hidden_layers) + 1):
        plt.text(i * 1.5, hidden_layers[i-1] / 2 - 0.5, f'Hidden Layer {i}\n(Feature Extraction)', fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    plt.text(len(hidden_layers) * 1.5 + 1.5, output_layer / 2 - 0.5, 'Output Layer\n(Sentiment Classification)', fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

    # Set limits and remove axes
    plt.xlim(-1, len(layers) * 1.5)
    plt.ylim(-1, max(input_layer, *hidden_layers, output_layer))
    plt.axis('off')
    plt.title('Neural Network Diagram for Sentiment Analysis on Amazon Book Reviews')
    plt.show()

plot_neural_network_contextual()
