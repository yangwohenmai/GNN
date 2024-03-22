import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.utils import to_undirected
import numpy as np
# Create random graph data
num_nodes = 10
num_features = 10
num_edges = 30

# Generate random node features and labels
features = torch.randn(num_nodes, num_features)
labels = torch.randint(0, 10, (num_nodes,))  # Assuming 10 classes for example

# Create edge indices
edge_index = torch.randint(0, num_nodes, (num_edges * 2,))
edge_index = edge_index.view(-1, 2)

# Ensure the graph is undirected
edge_index = to_undirected(edge_index)

# Create a Data object
data = Data(x=features, edge_index=edge_index, y=labels)

# Determine the number of classes from the unique labels
num_classes = torch.unique(labels).shape[0]

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.linear = Linear(out_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.linear(x)
        return x

# Instantiate the model
model = GCN(num_features, 16, num_classes).to('cpu')

# Define loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits, data.y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    predicted = logits.argmax(dim=1)
    correct = (predicted == data.y).sum().item()
    accuracy = 100 * correct / data.num_nodes
    print(f'Accuracy: {accuracy:.2f}%')






