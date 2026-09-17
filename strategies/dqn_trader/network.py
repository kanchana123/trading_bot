import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()

        # Convolutional layers for image processing (grayscale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)  # input is 320x320x1 (grayscale)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out((1, 320, 320))  # Adjusted for grayscale

        # Fully connected layers for image and numerical data processing
        self.fc_image = nn.Linear(conv_out_size, 512)
        self.fc_numerical = nn.Linear(17, 512)  # 17 numerical inputs
        self.fc_combined = nn.Linear(512 + 512, 512)
        self.fc_out = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        """Calculates the output size of the convolutional layers."""
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, image, numerical_data):
        """
        Forward pass of the DQN.

        Args:
            image (torch.Tensor): Image tensor of shape (batch_size, 1, 320, 320).
            numerical_data (torch.Tensor): Numerical data tensor of shape (batch_size, 17).

        Returns:
            torch.Tensor: Q-values for each action.
        """
        # Image processing
        conv_out = F.relu(self.conv1(image))
        conv_out = F.relu(self.conv2(conv_out))
        conv_out = F.relu(self.conv3(conv_out))
        conv_out = conv_out.view(conv_out.size()[0], -1)  # Flatten the output
        image_features = F.relu(self.fc_image(conv_out))

        # Numerical data processing
        numerical_features = F.relu(self.fc_numerical(numerical_data))

        # Concatenate image and numerical features
        combined_features = torch.cat((image_features, numerical_features), dim=1)

        # Output layer
        combined_features = F.relu(self.fc_combined(combined_features))
        q_values = self.fc_out(combined_features)

        return q_values

def visualize_network_weights(model, threshold=0.1, max_edge_width=5):
    """
    Visualizes the network graph, highlighting connections between neurons with thicker edges for larger weights.

    Args:
        model (nn.Module): The PyTorch model.
        threshold (float): The threshold for weight magnitude to consider.
        max_edge_width (float): The maximum width for the edges.
    """
    G = nx.DiGraph()
    layer_positions = {}
    layer_count = 0
    node_labels = {}  # Store node labels for better visualization

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_positions[name] = layer_count
            layer_count += 1

            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                
                # Add output nodes for this layer
                for i in range(out_channels):
                  node_name = f"{name}_out{i}"
                  if node_name not in G:
                    G.add_node(node_name, layer=name, layer_pos=layer_positions[name])
                    node_labels[node_name] = f"C{name[-1]}O{i}"

                # Add edges connecting neurons
                for i in range(out_channels):
                    for j in range(in_channels):
                        # Summarize weights from the entire kernel
                        kernel_weights = module.weight.data[i, j].view(-1)
                        weight_value = torch.mean(torch.abs(kernel_weights)).item()  # Mean absolute weight

                        node_from = f"{name}_in{j}"
                        if layer_positions[name] == 0 :
                          node_from = f"input_in{j}"
                          if node_from not in G:
                            G.add_node(node_from, layer="input" , layer_pos=-1)
                            node_labels[node_from] = f"I{j}"

                        node_to = f"{name}_out{i}"

                        if abs(weight_value) >= threshold:
                            G.add_edge(node_from, node_to, weight=weight_value)

            elif isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                
                # Add input and output nodes for this layer
                for j in range(in_features):
                    node_from = f"{name}_in{j}"
                    if node_from not in G:
                      G.add_node(node_from, layer=name, layer_pos=layer_positions[name]-1)
                      node_labels[node_from] = f"L{name[-1]}I{j}"

                for i in range(out_features):
                    node_to = f"{name}_out{i}"
                    if node_to not in G:
                      G.add_node(node_to, layer=name, layer_pos=layer_positions[name])
                      node_labels[node_to] = f"L{name[-1]}O{i}"

                # Add edges connecting neurons
                for i in range(out_features):
                    for j in range(in_features):
                        weight_value = module.weight.data[i, j].item()
                        node_from = f"{name}_in{j}"
                        node_to = f"{name}_out{i}"
                        if abs(weight_value) >= threshold:
                            G.add_edge(node_from, node_to, weight=weight_value)

    pos = nx.multipartite_layout(G, subset_key="layer_pos")

    # Calculate edge widths based on weight magnitude
    edge_widths = []
    for _, _, data in G.edges(data=True):
        weight = data["weight"]
        normalized_weight = min(abs(weight) / threshold, 1) if threshold != 0 else 0
        width = normalized_weight * max_edge_width
        edge_widths.append(width)

    edge_colors = [("red" if data["weight"] >= threshold else "blue") for _, _, data in G.edges(data=True)]
    node_colors = ["skyblue" if "in" in node else "lightgreen" for node in G.nodes()]
    
    # Draw the graph with labels
    plt.figure(figsize=(20, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,  # Show labels
        labels=node_labels,  # Use the created labels
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=500,  # Increase node size for label visibility
        width=edge_widths,
        font_size=8,
    )
    plt.title(f"Network Graph (Threshold = {threshold})")
    plt.show()

if __name__ == "__main__":
    num_actions = 3
    model = DQN(num_actions)

    # Example input data (grayscale)
    batch_size = 32
    image_input = torch.randn(batch_size, 1, 320, 320)  # Batch of 32 grayscale images
    numerical_input = torch.randn(batch_size, 17)  # Batch of 32 sets of numerical data

    # Forward pass
    q_values = model(image_input, numerical_input)
    print(q_values.shape)

    visualize_network_weights(model, threshold=0.001, max_edge_width=1)
