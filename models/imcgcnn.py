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
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.edge_dim = edge_dim
        self.normalize_batch = normalize_batch
        self.dropout = dropout
        self.bias = bias

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
        self.batch_norm = torch.nn.BatchNorm1d(self.out_channels) if self.normalize_batch else None
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

    

class EdgeUpdate(MessagePassing):
    """
    The Edge convolutional layer.
    Updates the edge features based on the node features.
    """
    def __init__(
        self,
        channels: Union[int, Tuple[int, int]],
        edge_dim: int = 0,
        normalize_batch: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.normalize_batch = normalize_batch
        self.dropout = dropout
        self.bias = bias
        self.edge_dim = edge_dim

        if isinstance(channels, int):
            self.in_channels, self.out_channels = channels, channels
        else:
            self.in_channels, self.out_channels = channels

        # in_dimension --> sum over all due to concatenation of inputs into z_ij := xi + xj + eij
        self.linear = Linear(self.in_channels + self.out_channels + self.edge_dim, self.out_channels, 
                                bias = self.bias, weight_initializer='glorot')
        self.batch_norm = torch.nn.BatchNorm1d(self.out_channels) if self.normalize_batch else None
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters() 
        if self.batch_norm is not None:
            self.batch_norm.reset_parameters()
    
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr = edge_attr, batch_norm = self.batch_norm)
        return edge_attr
    
    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor, batch_norm: bool) -> Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1) # now edge_attr are not optional
        #Old version with leaky relu
        #out = F.leaky_relu_(self.linear(z), negative_slope=0.01)
        out = F.sigmoid(self.linear(z))
        out = out if batch_norm is None else batch_norm(out)
        return out + edge_attr

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr

class IMcgcnn(torch.nn.Module):

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
        residual_connection: bool = False,
        dropout: float = 0.0,
        pre_conv_layers: int = 1,
        post_conv_layers: int = 2,
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
        self.residual_connection = residual_connection
        self.pre_conv_layers = pre_conv_layers
        self.post_conv_layers = post_conv_layers


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
        self.edge_convs = torch.nn.ModuleList()
        self.norm_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.node_convs.append(NodeConv(channels = hidden_channels, edge_dim=hidden_channels,dropout=dropout)) 
            self.edge_convs.append(EdgeUpdate(channels = hidden_channels, edge_dim=hidden_channels,dropout=dropout))
            self.norm_convs.append(BatchNorm(hidden_channels))

        # Output layers
        self.pre_pool_layer = Linear(hidden_channels, hidden_channels)
        #self.out_edges = Linear(hidden_channels, hidden_channels)

        self.post_conv_hidden = torch.nn.ModuleList()
        for _ in range(self.post_conv_layers):
            self.post_conv_hidden.append(Linear(hidden_channels, hidden_channels))
        self.output_layer = Linear(hidden_channels, out_channels)
        
        self.reset_parameters()


    def reset_parameters(self):
        
        for pre_conv_node, pre_conv_edge in zip(self.pre_conv_nodes, self.pre_conv_edges):
            pre_conv_node.reset_parameters()
            pre_conv_edge.reset_parameters()

        for node_conv, edge_conv in zip(self.node_convs, self.edge_convs):
            node_conv.reset_parameters()
            edge_conv.reset_parameters()
        self.pre_pool_layer.reset_parameters()
        #self.out_edges.reset_parameters()
        for hidden_layer in self.post_conv_hidden:
            hidden_layer.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: OptTensor, batch: Tensor) -> Tensor:
        """
        """

        # Setup input features
        for pre_node, pre_edge in zip(self.pre_conv_nodes, self.pre_conv_edges):
            x = getattr(F, self.activation)(pre_node(x))
            edge_attr = getattr(F, self.activation)(pre_edge(edge_attr))

        # Graphs convolutions
        for node_conv, edge_conv, norm_conv in zip(self.node_convs, self.edge_convs, self.norm_convs):
            # we follow the skip connection (Res+) strategy
            # Normalization -> Activation -> Dropout -> Conv -> Res
            # Conv :  g_l := Conv(x_l)
            # Res : x_l = g_{l-1} + x_{l-1}
            if self.residual_connection:
                #norm
                x_in = norm_conv(x)
                edge_attr_in = norm_conv(edge_attr)
                #activation
                x_in = getattr(F, self.activation)(x_in)
                edge_attr_in = getattr(F, self.activation)(edge_attr_in)
                #dropout
                x_in = F.dropout(x_in, p=self.dropout, training=self.training)
                edge_attr_in = F.dropout(edge_attr_in, p=self.dropout, training=self.training)
                #conv
                x_in = node_conv(x_in, edge_index, edge_attr)
                edge_attr_in = edge_conv(x_in, edge_index, edge_attr)
                #res
                x = x_in + x
                edge_attr = edge_attr_in + edge_attr
            else:
                x = node_conv(x, edge_index, edge_attr)
                edge_attr = edge_conv(x, edge_index, edge_attr)
                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        # pooling setup --> Only for nodes
        x = getattr(F, self.activation)(self.pre_pool_layer(x))
        #edge_attr = getattr(F, self.activation)(self.out_edges(edge_attr))
        # Pooling
        x = getattr(torch_geometric.nn, self.pooling)(x, batch)
        #x = F.dropout(x, p=self.dropout, training=self.training)
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
            "residual_connection" : self.residual_connection,
            "pre_conv_layers" : self.pre_conv_layers,
            "post_conv_layers" : self.post_conv_layers,
            "dropout" : self.dropout,
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
                f'residual_connection={self.residual_connection},'
                f'pre_conv_layers={self.pre_conv_layers},'
                f'post_conv_layers={self.post_conv_layers},'
                f'dropout={self.dropout:.2f},'
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
            f"    residual_connection = {self.residual_connection},\n"
            f"    pre_conv_layers     = {self.pre_conv_layers},\n"
            f"    post_conv_layers    = {self.post_conv_layers},\n"
            f"    dropout             = {self.dropout:.2f}\n"
            f")"
        )

