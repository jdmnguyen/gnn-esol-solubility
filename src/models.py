# src/models.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class SolubilityGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.act = nn.ReLU()

        # MLP head
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)

        # Pool atom features to get molecule representation
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        # MLP head
        x = self.act(self.lin1(x))
        out = self.lin2(x)  # [num_graphs, 1]
        return out.view(-1)  # flatten
