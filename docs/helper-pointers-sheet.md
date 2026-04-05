CSCI 6850 Project Hints
The following is an example of the forward method defined for the graph machine learning model. You can modify it to get your version.

import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv

def forward(self, batch_data):
    # Note: batch_data is provided by DataLoader
    x, edge_index = batch_data.x, batch_data.edge_index

    x = self.conv1(x, edge_index)
    x = self.batch_norm1(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.2, training=self.training)

    for gcnconv, batch_norm in zip(self.gcnConvs, self.batch_norms):
        x = batch_norm(gcnconv(x, edge_index))
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

    out = self.post_mlp(x)
    return out


CSCI 6850 Project Hints – Notes
Note:

conv_1 and gcnconv are instances of torch_geometric.nn.GCNConv.
post_mlp is an MLP for prediction. Its input and output dimension should be the output dimension of the gcnconv in front of it and one, respectively.

Example of post_mlp:
Assuming the output dimension of the gcnconv layer in front of the MLP is out_channels, then:
self.post_mlp = Sequential(
    Linear(self.out_channels, self.out_channels // 2),
    ReLU(),
    Linear(self.out_channels // 2, 1)
)
CSCI 6850 Project Hints – Suggested Workflow
The following is a suggested flow of steps:

PyG dataset generation and splitting
Dataloader set up
Model set up
Set up training optimizer and learning rate scheduler
Model training, validating and testing