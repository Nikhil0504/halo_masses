import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv


class SAGEGraphConvNet(nn.Module):
    """A simple GNN built using SAGEConv layers.
    """
    def __init__(self, n_in=3, n_hidden=256, n_out=1):
        super(SAGEGraphConvNet, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.conv1 = SAGEConv(self.n_in, self.n_hidden)
        self.conv2 = SAGEConv(self.n_hidden, self.n_hidden)
        # self.fc = nn.Linear(n_hidden, n_out, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_in + 2 * self.n_hidden, self.n_hidden, bias=True),
            nn.ReLU(),
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, 2*self.n_out, bias=True)
        )

    def forward(self, data):
        x0, edge_index = data.x_hydro, data.edge_index

        x1 = self.conv1(x0, edge_index)
        x2 = self.conv2(F.relu(x1), edge_index)
        mlp_out = self.mlp(torch.cat([x0, F.relu(x1), F.relu(x2)], dim=-1))
        return mlp_out


class EdgeInteractionLayer(MessagePassing):
    """Graph interaction layer that combines node & edge features on edges.
    """
    def __init__(self, n_in, n_hidden, n_latent, aggr='sum', act_fn=nn.SiLU):
        super(EdgeInteractionLayer, self).__init__(aggr)

        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            act_fn(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            act_fn(),
            nn.Linear(n_hidden, n_latent, bias=True),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(inputs)

class EdgeInteractionGNN(nn.Module):
    """Graph net over nodes and edges with multiple unshared layers, and sequential layers with residual connections.
    Self-loops also get their own MLP (i.e. galaxy-halo connection).
    """
    def __init__(self, n_layers, node_features=2, edge_features=6, hidden_channels=64, aggr="sum", latent_channels=64, n_out=1, n_unshared_layers=4, act_fn=nn.SiLU):
        super(EdgeInteractionGNN, self).__init__()

        self.n_in = 2 * node_features + edge_features 
        self.n_out = n_out
        self.n_pool = (len(aggr) if isinstance(aggr, list) else 1) 
        
        layers = [
            nn.ModuleList([
                EdgeInteractionLayer(self.n_in, hidden_channels, latent_channels, aggr=aggr, act_fn=act_fn)
                for _ in range(n_unshared_layers)
            ])
        ]
        for _ in range(n_layers-1):
            layers += [
                nn.ModuleList([
                    EdgeInteractionLayer(self.n_pool * (2 * latent_channels * n_unshared_layers + edge_features), hidden_channels, latent_channels, aggr=aggr, act_fn=act_fn) 
                    for _ in range(n_unshared_layers)
                ])
            ]
   
        self.layers = nn.ModuleList(layers)
        
        self.galaxy_environment_mlp = nn.Sequential(
            nn.Linear(self.n_pool * n_unshared_layers * latent_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, latent_channels, bias=True)
        )

        self.galaxy_halo_mlp = nn.Sequential(
            nn.Linear(node_features, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, latent_channels, bias=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * latent_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, 2 * n_out, bias=True)
        )
    
    def forward(self, data):
        
        # determine edges by getting neighbors within radius defined by `D_link`
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # update hidden state on edge (h, or sometimes e_ij in the text)
        h = torch.cat(
            [
                unshared_layer(data.x_hydro, edge_index=edge_index, edge_attr=edge_attr)
                for unshared_layer in self.layers[0]
            ], 
            axis=1
        )
        
        for layer in self.layers[1:]:
            # if multiple layers deep, also use a residual layer
            h += torch.cat(
                [
                    unshared_layer(h, edge_index=edge_index, edge_attr=edge_attr) 
                    for unshared_layer in layer
                ], 
                axis=1
            )
                        
        out =  torch.cat([self.galaxy_environment_mlp(h), self.galaxy_halo_mlp(data.x_hydro)], axis=1)
        return self.fc(out)
