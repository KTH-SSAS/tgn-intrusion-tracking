import torch
from torch_geometric.data import Data
from torch_geometric_temporal.nn import TGNMemory, TemporalConv
import torch_geometric_temporal

def main():
    print("PyTorch Version:", torch.__version__)
    print("PyTorch Geometric Version:", torch_geometric_temporal.__version__)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch will use the GPU.")
    else:
        print("CUDA is not available. PyTorch will use the CPU.")

    # Create a simple graph data object
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    print("Graph Data:", data)

    # Initialize a simple Temporal Conv layer
    try:
        conv = TemporalConv(
            node_features=1,
            edge_features=1,
            memory=10,
            hidden_channels=16
        )
        print("TemporalConv layer initialized successfully.")
    except Exception as e:
        print("Error initializing TemporalConv layer:", e)

if __name__ == "__main__":
    main()
