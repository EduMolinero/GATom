import time
from typing import Tuple, Union

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GATv2Conv, SuperGATConv, GCNConv, BatchNorm, GraphNorm, DiffGroupNorm
from torch_geometric.loader import DataLoader
from typing import Optional
                    
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter


from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor, PairOptTensor, Tensor

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import (
    add_self_loops,
    batched_negative_sampling,
    dropout_edge,
    is_undirected,
    negative_sampling,
    remove_self_loops,
    softmax,
    to_undirected,
)


class NodeConv(MessagePassing):
    """
    Update the node features based on the edge features plus edge updates.
    It is based on CGConv (Crystal graph prl) edge updates based on ALIGNN.
    """
    def __init__(
        self,
        channels: Union[int, Tuple[int, int]],
        edge_dim: int = 0,
        normalize_batch: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        batch_track_stats: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.edge_dim = edge_dim
        self.normalize_batch = normalize_batch
        self.dropout = dropout
        self.bias = bias
        self.batch_track_stats = batch_track_stats

        if isinstance(channels, int):
            self.in_channels, self.out_channels = channels, channels
        else:
            self.in_channels, self.out_channels = channels
        
        # in_dimension --> sum over all due to concatenation of inputs into z_ij := xi + xj + eij
        self.lin_conv = Linear(self.in_channels + self.out_channels + self.edge_dim, self.out_channels, 
                                bias = self.bias, weight_initializer='glorot')
        self.lin_softmax = Linear(self.in_channels + self.out_channels + self.edge_dim, self.out_channels, 
                                bias = self.bias, weight_initializer='glorot')

        self.linear_edge = Linear(self.in_channels + self.out_channels + self.edge_dim, self.out_channels, 
                                  bias = self.bias, weight_initializer='glorot')
        self.batch_norm = torch.nn.BatchNorm1d(self.out_channels, track_running_stats=self.batch_track_stats) if self.normalize_batch else None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_conv.reset_parameters()
        self.lin_softmax.reset_parameters()
        if self.batch_norm is not None:
            self.batch_norm.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out if self.batch_norm is None else self.batch_norm(out)
        return out + x
    
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1) if edge_attr is not None else torch.cat([x_i, x_j], dim=-1)
        #Old version with leaky relu
        #return F.softplus(self.lin_softmax(z)) * F.leaky_relu_(self.lin_conv(z), negative_slope=0.01)
        return F.softplus(self.lin_softmax(z)) * F.sigmoid(self.lin_conv(z))
    
class cgcnn(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        activation: str = "relu",
        pooling: str = "global_mean_pool",
        task: str = "regression",
        dropout: float = 0.0,
        pre_conv_layers: int = 1,
        post_conv_layers: int = 2,
        batch_track_stats: bool = True,
    ):
        
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.task = task
        self.pooling = pooling
        self.pre_conv_layers = pre_conv_layers
        self.post_conv_layers = post_conv_layers
        self.batch_track_stats = batch_track_stats


        # We make them have the same hidden channels so we can pool them together more easily.
        self.pre_conv_nodes = torch.nn.ModuleList()
        self.pre_conv_edges = torch.nn.ModuleList()
        for i in range(self.pre_conv_layers):
            if i == 0:
                self.pre_conv_nodes.append(Linear(in_channels, hidden_channels))
                self.pre_conv_edges.append(Linear(edge_dim, hidden_channels))
            else:
                self.pre_conv_nodes.append(Linear(hidden_channels, hidden_channels))
                self.pre_conv_edges.append(Linear(hidden_channels, hidden_channels))

        self.node_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.node_convs.append(NodeConv(
                                            channels = hidden_channels, 
                                            edge_dim=hidden_channels,
                                            dropout=dropout, 
                                            batch_track_stats=self.batch_track_stats
                                )) 
        # Output layers
        self.pre_pool_layer = Linear(hidden_channels, hidden_channels)

        self.post_conv_hidden = torch.nn.ModuleList()
        for _ in range(self.post_conv_layers):
            self.post_conv_hidden.append(Linear(hidden_channels, hidden_channels))
        self.output_layer = Linear(hidden_channels, out_channels)
        
    #     self.reset_parameters()


    # def reset_parameters(self):
    #     for pre_conv_node in self.pre_conv_nodes:
    #         pre_conv_node.reset_parameters()
    #     for node_conv in self.node_conv:
    #         node_conv.reset_parameters()
    #     self.pre_pool_layer.reset_parameters()
    #     for hidden_layer in self.post_conv_hidden:
    #         hidden_layer.reset_parameters()
    #     self.output_layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: OptTensor, batch: Tensor) -> Tensor:
        """
        """

        # Setup input features
        for pre_node, pre_edge in zip(self.pre_conv_nodes, self.pre_conv_edges):
            x = getattr(F, self.activation)(pre_node(x))
            edge_attr = getattr(F, self.activation)(pre_edge(edge_attr))

        # Graphs convolutions
        for node_conv in self.node_convs:
            x = node_conv(x, edge_index, edge_attr)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # pooling setup --> Only for nodes
        x = getattr(F, self.activation)(self.pre_pool_layer(x))
        # Pooling
        x = getattr(torch_geometric.nn, self.pooling)(x, batch)
        # Final layers
        for hidden_layer in self.post_conv_hidden:
            x = getattr(F, self.activation)(hidden_layer(x))
            #x = F.dropout(x, p=self.dropout, training=self.training)

        # Predictor:
        if self.task == 'classification':
            x = F.log_softmax(self.output_layer(x), dim=self.out_channels)
        elif self.task == 'regression':
            x = self.output_layer(x)
        else:
            raise ValueError('Task not recognized.')
        
        if x.shape[1] == 1:
            return x.view(-1)
        else:
            return x

    
    def _dict_model(self) -> dict:
        """
        Returns a dict with the parameters of the model without the parameters of torch.nn.Module
        """
        return {
            "in_channels" : self.in_channels,
            "hidden_channels" : self.hidden_channels,
            "out_channels" : self.out_channels,
            "edge_dim" : self.edge_dim,
            "num_layers" : self.num_layers,
            "activation" : self.activation,
            "pooling" : self.pooling,
            "task" : self.task,
            "pre_conv_layers" : self.pre_conv_layers,
            "post_conv_layers" : self.post_conv_layers,
            "dropout" : self.dropout,
            "batch_track_stats" : self.batch_track_stats
        }
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers},'
                f'activation={self.activation},'
                f'pooling={self.pooling},'
                f'task={self.task},'
                f'pre_conv_layers={self.pre_conv_layers},'
                f'post_conv_layers={self.post_conv_layers},'
                f'dropout={self.dropout:.2f},'
                f'batch_track_stats={self.batch_track_stats},'
                f')')
    
    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"    in_channels         = {self.in_channels},\n"
            f"    hidden_channels     = {self.hidden_channels},\n"
            f"    out_channels        = {self.out_channels},\n"
            f"    edge_dim            = {self.edge_dim},\n"
            f"    num_layers          = {self.num_layers},\n"
            f"    activation          = {self.activation},\n"
            f"    pooling             = {self.pooling},\n"
            f"    task                = {self.task},\n"
            f"    pre_conv_layers     = {self.pre_conv_layers},\n"
            f"    post_conv_layers    = {self.post_conv_layers},\n"
            f"    dropout             = {self.dropout:.2f}\n"
            f"    batch_track_stats   = {self.batch_track_stats}\n"
            f")"
        )