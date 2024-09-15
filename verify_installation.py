# Verify torch-geometric-temporal installation
import torch
import numpy as np  # Add NumPy
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal

# Create a simple synthetic dataset
node_features = [torch.rand(5, 4) for _ in range(10)]  # 10 time steps, 5 nodes, 4 features
edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Edge indices
edge_weights = torch.rand(4)  # Weights for the edges

# Change targets to NumPy arrays instead of PyTorch tensors
targets = [np.random.rand(5).astype(np.float32) for _ in range(10)]  # Targets for each time step

# Create a StaticGraphTemporalSignal object
dataset = StaticGraphTemporalSignal(edge_index=edges, edge_weight=edge_weights, features=node_features, targets=targets)

# Split the dataset into train and test sets
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# Create a simple model using GConvGRU
model = GConvGRU(in_channels=4, out_channels=2, K=2)

# Perform a forward pass on one snapshot of the dataset
snapshot = next(iter(train_dataset))
x, edge_index, edge_weight = snapshot.x, snapshot.edge_index, snapshot.edge_weight

# Forward pass
out = model(x, edge_index, edge_weight)
print("Output shape:", out.shape)
