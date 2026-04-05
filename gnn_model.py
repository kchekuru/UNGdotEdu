import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv

class ChargeGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=192, num_layers=4, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.batch_norm1 = BatchNorm1d(hidden_channels)

        self.gcnConvs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # conv1 counts as one layer; add remaining graph conv blocks.
        for _ in range(max(num_layers - 1, 0)):
            self.gcnConvs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        self.post_mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Linear(hidden_channels // 2, 1)
        )

    def forward(self, batch_data):
        x, edge_index = batch_data.x, batch_data.edge_index
        edge_weight = None

        # If geometry is available, use distance-derived edge weights for GCN.
        if hasattr(batch_data, "edge_attr") and batch_data.edge_attr is not None:
            edge_attr = batch_data.edge_attr
            if edge_attr.dim() == 3 and edge_attr.shape[1] == 1:
                edge_attr = edge_attr.squeeze(1)

            if edge_attr.dim() == 2 and edge_attr.shape[1] > 1:
                distances = torch.linalg.norm(edge_attr, dim=-1)
            else:
                distances = edge_attr.view(-1)

            edge_weight = 1.0 / (1.0 + distances)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for gcnconv, batch_norm in zip(self.gcnConvs, self.batch_norms):
            x = gcnconv(x, edge_index, edge_weight=edge_weight)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.post_mlp(x)
        return out