import time
from typing import Tuple, Union

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GATv2Conv, SuperGATConv, GCNConv, BatchNorm, GraphNorm, DiffGroupNorm
from torch_geometric.loader import DataLoader
from typing import Optional
                    
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from .mgu import liGRUCell, MGUCell

import torch_geometric
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor, PairOptTensor, Tensor
from torch_geometric.nn import aggr

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


recurrent_cell_classes = {
    'gru': GRUCell,
    'ligru': liGRUCell,
    'mgu' : MGUCell,    
}

                    
class GATom(torch.nn.Module):
    r"""[[[[[ ADD DESCRIPTION ]]]]]]

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        layers_attention: int,
        num_timesteps: int,
        recurrent_cell: str = 'gru',
        pooling: str = 'global_add_pool',
        aggregation: str = 'add',
        activation: str = 'relu',
        heads: int = 1,
        residual_connection: bool = False,
        task: str = 'regression',
        dropout: float = 0.0,
        pre_conv_layers: int = 1,
        post_conv_layers: int = 2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim 
        self.layers_attention = layers_attention
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.heads = heads
        self.res_connection = residual_connection
        self.task = task
        self.pooling = pooling
        self.activation = activation 
        self.recurrent_cell_class = recurrent_cell_classes.get(recurrent_cell.lower())
        self.pre_conv_layers = pre_conv_layers
        self.post_conv_layers = post_conv_layers


        # check for aggregation with learnable parameters
        if aggregation == 'softmax' or aggregation == 'powermean':
            if aggregation == 'softmax':
                self.aggregation = aggr.SoftmaxAggregation(learn=True)
            elif aggregation == 'powermean':
                self.aggregation = aggr.PowerMeanAggregation(learn=True)
        else:
            self.aggregation = aggregation

        #################################################################
        # Embedding block
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

        #################################################################
        # Pocessing block
        # the edge and node convolutions
        # self.node_convs = torch.nn.ModuleList()
        # self.edge_convs = torch.nn.ModuleList()

        # for _ in range(self.layers_embedding - 1):
        #     self.node_convs.append(NodeConv(channels = hidden_channels, edge_dim=hidden_channels,dropout=dropout)) 
        #     self.edge_convs.append(EdgeUpdate(channels = hidden_channels, edge_dim=hidden_channels,dropout=dropout))



        # attention mechanism
        # Local attention
        self.embedding_att_convs = torch.nn.ModuleList()
        self.embedding_grus = torch.nn.ModuleList()
        self.embedding_norms = torch.nn.ModuleList()
        # right now we only have implemented the attention mechanism on the node features.
        # To do: implement the attention mechanism on the edge features.
        for _ in range(self.layers_attention - 1):
            
            self.embedding_att_convs.append(GATv2Conv(hidden_channels, hidden_channels, heads = self.heads, dropout=dropout,
                                            add_self_loops=False, negative_slope=0.01, edge_dim=hidden_channels,
                                            aggr = self.aggregation))
            # The output of the GATv2Conv has dimensions (N, out_channels * heads)
            # So we need to properly set up the dimensions of the h state in the GRUCell
            self.embedding_grus.append(self.recurrent_cell_class(hidden_channels * self.heads, hidden_channels))
            if self.recurrent_cell_class is None:
                raise ValueError(f"Unknown recurrent cell: {self.recurrent_cell}")
            
            #self.embedding_norms.append(BatchNorm(hidden_channels))
            self.embedding_norms.append(DiffGroupNorm(hidden_channels, groups=10))

        # Global attention
        # In order to have a global attention we need to add a super node to the graph
        # We add a super node to the graph with only one head of attention
        # otherwise we this leads to problems with the global pooling
        self.global_conv = GATv2Conv(hidden_channels, hidden_channels, heads = 1, dropout=dropout,
                                    add_self_loops=False, negative_slope=0.01, edge_dim=hidden_channels,
                                    aggr = self.aggregation)
        self.global_gru = self.recurrent_cell_class(hidden_channels, hidden_channels)
        #self.global_norm = BatchNorm(hidden_channels)
        self.global_norm = DiffGroupNorm(hidden_channels, groups=10)

        #################################################################
        # Readout block
        self.post_conv_hidden = torch.nn.ModuleList()
        for _ in range(self.post_conv_layers):
            self.post_conv_hidden.append(Linear(hidden_channels, hidden_channels))

        self.output_layer = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""

        for pre_conv_node, pre_conv_edge in zip(self.pre_conv_nodes, self.pre_conv_edges):
            pre_conv_node.reset_parameters()
            pre_conv_edge.reset_parameters()
        # for node_conv, edge_conv in zip(self.node_convs, self.edge_convs):
        #     node_conv.reset_parameters()
        #     edge_conv.reset_parameters()
        for att_conv, gru, norm in zip(self.embedding_att_convs, self.embedding_grus, self.embedding_norms):
            att_conv.reset_parameters()
            gru.reset_parameters()
            norm.reset_parameters()
        for hidden_layer in self.post_conv_hidden:
            hidden_layer.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: OptTensor,
                batch: Tensor) -> Tensor:
        """
        """
        #Embedding block
        for pre_node, pre_edge in zip(self.pre_conv_nodes, self.pre_conv_edges):
            x = getattr(F, self.activation)(pre_node(x))
            edge_attr = getattr(F, self.activation)(pre_edge(edge_attr))

        #Processing block
        # for node_conv, edge_conv in zip(self.node_convs, self.edge_convs):
        #     #Update node features
        #     x = node_conv(x, edge_index, edge_attr)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #     #Update edge features
        #     edge_attr = edge_conv(x, edge_index, edge_attr)
        #     edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        # attention mechanism
        # Local attention
        for index_layer, (att_conv, gru, norm) in enumerate(zip(self.embedding_att_convs, self.embedding_grus, self.embedding_norms)):
            # we follow the skip connection (Res+) strategy
            # Normalization -> Activation -> Dropout -> Conv -> Res
            # Conv :  GAT + GRU -> g_l := Conv(x_l)
            # Res : x_l = g_{l-1} + x_{l-1}
            if self.res_connection:
                #x_old = x if index_layer == 0 else None
                ## Res+ connection
                g = norm(x)
                g = F.relu(g)
                g = F.dropout(g, p=self.dropout, training=self.training)
                for t in range(self.num_timesteps):
                    h = F.elu(att_conv(g, edge_index, edge_attr))
                    g = gru(h, g).relu()
                x = x + g
                ## Res connection
                # Conv -> Norm -> activation -> Res
                # for t in range(self.num_timesteps):
                #     h = F.elu(att_conv(x, edge_index, edge_attr))
                #     h = F.dropout(h, p=self.dropout, training=self.training)
                #     g = gru(h, x).relu()
                # g = norm(g)
                # x = x + F.relu(g)
            else:
                for t in range(self.num_timesteps):
                    h = F.elu(att_conv(x, edge_index, edge_attr))
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    x = gru(h, x).relu()
                x = norm(x)

        # Global attention
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)
        # Additive pooling because we are adding a super node outside the graph
        out = getattr(torch_geometric.nn, self.pooling)(x, batch).relu()
        for t in range(self.num_timesteps):
            h = F.elu(self.global_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.global_gru(h, out).relu()

        x = self.global_norm(out)

        # Readout block
        # Mean pooling because now we want to make predictions
        #x = global_mean_pool(x, batch).relu_()
        #x = F.dropout(x, p=self.dropout, training=self.training)
        for hidden_layer in self.post_conv_hidden:
            x = getattr(F, self.activation)(hidden_layer(x))
            #x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Predictor:
        if self.task == 'classification':
            x = F.sigmoid(self.output_layer(x))
        elif self.task == 'regression':
            x = self.output_layer(x)
        else:
            raise ValueError('Task not recognized.')
    
        if x.shape[1] == 1:
            return x.view(-1)
        else:
            return x


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'layers_attention={self.layers_attention},'
                f'pre_conv_layers={self.pre_conv_layers},'
                f'post_conv_layers={self.post_conv_layers},'
                f'num_timesteps={self.num_timesteps},'
                f'aggregation={self.aggregation},'
                f'heads={self.heads},'
                f'residual_connection={self.res_connection},'
                f'dropout={self.dropout:.2f},'
                f'task={self.task},'
                f'pooling={self.pooling},'
                f'activation={self.activation},'
                f'recurrent_cell={self.recurrent_cell_class.__name__}'
                f')')
    
    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"    in_channels          = {self.in_channels},\n"
            f"    hidden_channels      = {self.hidden_channels},\n"
            f"    out_channels         = {self.out_channels},\n"
            f"    edge_dim             = {self.edge_dim},\n"
            f"    layers_attention     = {self.layers_attention},\n"
            f"    pre_conv_layers      = {self.pre_conv_layers},\n"
            f"    post_conv_layers     = {self.post_conv_layers},\n"
            f"    num_timesteps        = {self.num_timesteps},\n"
            f"    aggregation          = {self.aggregation},\n"
            f"    heads                = {self.heads},\n"
            f"    residual_connection  = {self.res_connection},\n"
            f"    dropout              = {self.dropout:.2f},\n"
            f"    task                 = {self.task},\n"
            f"    pooling              = {self.pooling},\n"
            f"    activation           = {self.activation},\n"
            f"    recurrent_cell       = {self.recurrent_cell_class.__name__}\n"
            f")"
        )


    
    def _dict_model(self) -> dict:
        """
        Returns a dict with the parameters of the model without the parameters of torch.nn.Module
        """
        return {
            "in_channels" : self.in_channels,
            "hidden_channels" : self.hidden_channels,
            "out_channels" : self.out_channels,
            "edge_dim" : self.edge_dim,
            "layers_attention" : self.layers_attention,
            "pre_conv_layers" : self.pre_conv_layers,
            "post_conv_layers" : self.post_conv_layers,
            "num_timesteps" : self.num_timesteps,
            "aggregation" : self.aggregation,
            "heads" : self.heads,
            "res_connection" : self.res_connection,
            "dropout" : self.dropout,
            "task" : self.task,
            "pooling" : self.pooling,
            "activation" : self.activation,
            "recurrent_cell" : self.recurrent_cell_class.__name__
        }

                    
                    

####################################################################################################
# OLD CODE
####################################################################################################


# class NodeConv(MessagePassing):
#     """
#     Update the node features based on the edge features.
#     It is based on CGConv (Crystal graph prl)
#     """
#     def __init__(
#         self,
#         channels: Union[int, Tuple[int, int]],
#         edge_dim: int = 0,
#         normalize_batch: bool = True,
#         dropout: float = 0.0,
#         bias: bool = True,
#         **kwargs,
#     ):
#         super().__init__(aggr="add", **kwargs)
#         self.edge_dim = edge_dim
#         self.normalize_batch = normalize_batch
#         self.dropout = dropout
#         self.bias = bias

#         if isinstance(channels, int):
#             self.in_channels, self.out_channels = channels, channels
#         else:
#             self.in_channels, self.out_channels = channels
        
#         # in_dimension --> sum over all due to concatenation of inputs into z_ij := xi + xj + eij
#         self.lin_conv = Linear(self.in_channels + self.out_channels + self.edge_dim, self.out_channels, 
#                                 bias = self.bias, weight_initializer='glorot')
#         self.lin_softmax = Linear(self.in_channels + self.out_channels + self.edge_dim, self.out_channels, 
#                                 bias = self.bias, weight_initializer='glorot')
#         self.batch_norm = torch.nn.BatchNorm1d(self.out_channels) if self.normalize_batch else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_conv.reset_parameters()
#         self.lin_softmax.reset_parameters()
#         if self.batch_norm is not None:
#             self.batch_norm.reset_parameters()

#     def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor) -> Tensor:
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
#         out = out if self.batch_norm is None else self.batch_norm(out)
#         return out + x
    
#     def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
#         z = torch.cat([x_i, x_j, edge_attr], dim=-1) if edge_attr is not None else torch.cat([x_i, x_j], dim=-1)
#         return F.softplus(self.lin_softmax(z)) * F.leaky_relu_(self.lin_conv(z), negative_slope=0.01)

    

# class EdgeUpdate(MessagePassing):
#     """
#     The Edge convolutional layer.
#     Updates the edge features based on the node features.
#     """
#     def __init__(
#         self,
#         channels: Union[int, Tuple[int, int]],
#         edge_dim: int = 0,
#         normalize_batch: bool = True,
#         dropout: float = 0.0,
#         bias: bool = True,
#         **kwargs,
#     ):
#         super().__init__(aggr="add", **kwargs)
#         self.normalize_batch = normalize_batch
#         self.dropout = dropout
#         self.bias = bias
#         self.edge_dim = edge_dim

#         if isinstance(channels, int):
#             self.in_channels, self.out_channels = channels, channels
#         else:
#             self.in_channels, self.out_channels = channels

#         # in_dimension --> sum over all due to concatenation of inputs into z_ij := xi + xj + eij
#         self.linear = Linear(self.in_channels + self.out_channels + self.edge_dim, self.out_channels, 
#                                 bias = self.bias, weight_initializer='glorot')
#         self.batch_norm = torch.nn.BatchNorm1d(self.out_channels) if self.normalize_batch else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.linear.reset_parameters() 
#         if self.batch_norm is not None:
#             self.batch_norm.reset_parameters()
    
#     def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
#         edge_attr = self.edge_updater(edge_index, x=x, edge_attr = edge_attr, batch_norm = self.batch_norm)
#         return edge_attr
    
#     def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor, batch_norm: bool) -> Tensor:
#         z = torch.cat([x_i, x_j, edge_attr], dim=-1) # now edge_attr are not optional
#         out = F.leaky_relu_(self.linear(z), negative_slope=0.01)
#         out = out if batch_norm is None else batch_norm(out)
#         return out

#     def message(self, edge_attr: Tensor) -> Tensor:
#         return edge_attr
    
# class GATom(torch.nn.Module):
#     r"""[[[[[ ADD DESCRIPTION ]]]]]]

#     Args:
#         in_channels (int): Size of each input sample.
#         hidden_channels (int): Hidden node feature dimensionality.
#         out_channels (int): Size of each output sample.
#         edge_dim (int): Edge feature dimensionality.
#         num_layers (int): Number of GNN layers.
#         num_timesteps (int): Number of iterative refinement steps for global
#             readout.
#         dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

#     """
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         out_channels: int,
#         edge_dim: int,
#         num_layers: int,
#         num_timesteps: int,
#         attention_type: str,
#         dropout: float = 0.0,
#     ):
#         super().__init__()

#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.edge_dim = edge_dim 
#         self.num_layers = num_layers
#         self.num_timesteps = num_timesteps
#         self.dropout = dropout
#         self.attention_type = attention_type

#         self.lin1 = Linear(in_channels, hidden_channels)

#         # We first convolute the edge features into the nodes features
#         self.edge_convs = torch.nn.ModuleList()
#         self.edge_grus = torch.nn.ModuleList()

#         for _ in range(num_layers - 1):
#             conv = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim,
#                             dropout=dropout, add_self_loops=True, negative_slope=0.01)
            
#             self.edge_convs.append(conv)
#             self.edge_grus.append(GRUCell(hidden_channels, hidden_channels))

#         self.atom_convs = torch.nn.ModuleList()
#         self.atom_grus = torch.nn.ModuleList()

#         for _ in range(num_layers - 1):
#             if self.attention_type == 'GAT':
#                 conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
#                             add_self_loops=True, negative_slope=0.01)
#             elif self.attention_type == 'GATv2':
#                 conv = GATv2Conv(hidden_channels, hidden_channels, dropout=dropout,
#                             add_self_loops=True, negative_slope=0.01)

#             else:
#                 raise ValueError('Attention type not recognized.')

#             self.atom_convs.append(conv)
#             self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

#         # We cannot use SuperGATConv for the global pooling layer.
#         # it doesn't make sense (I think), but check it in the future.
#         self.crystal_conv = GATv2Conv(hidden_channels, hidden_channels,
#                             dropout=dropout, add_self_loops=True,
#                             negative_slope=0.01)

#         self.crystal_conv.explain = False  # Cannot explain global pooling.
#         self.crystal_gru = GRUCell(hidden_channels, hidden_channels)

#         self.lin2 = Linear(hidden_channels, out_channels)

#         self.reset_parameters()

#     def reset_parameters(self):
#         r"""Resets all learnable parameters of the module."""
#         self.lin1.reset_parameters()
#         for conv, gru in zip(self.edge_convs, self.edge_grus):
#             conv.reset_parameters()
#             gru.reset_parameters()
#         for conv, gru in zip(self.atom_convs, self.atom_grus):
#             conv.reset_parameters()
#             gru.reset_parameters()
#         self.crystal_conv.reset_parameters()
#         self.crystal_gru.reset_parameters()
#         self.lin2.reset_parameters()

#     def forward(self, x: Tensor, edge_index: Tensor, edge_attr: OptTensor,
#                 batch: Tensor) -> Tensor:
#         """"""  # noqa: D419
#         #Edge embedding into Atom :
#         x = F.leaky_relu_(self.lin1(x))
#         for conv, gru in zip(self.edge_convs, self.edge_grus):
#             h = conv(x, edge_index, edge_attr)
#             h = F.elu(h)
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             x = gru(h, x).relu()

#         # Atom Embedding:
#         for conv, gru in zip(self.atom_convs, self.atom_grus):
#             h = conv(x, edge_index)
#             h = F.elu(h)
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             x = gru(h, x).relu()

#         # Crystal Embedding:
#         row = torch.arange(batch.size(0), device=batch.device)
#         edge_index = torch.stack([row, batch], dim=0)

#         out = global_add_pool(x, batch).relu_()
#         for t in range(self.num_timesteps):
#             h = F.elu_(self.crystal_conv((x, out), edge_index))
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             out = self.crystal_gru(h, out).relu_()

#         # Predictor:
#         out = F.dropout(out, p=self.dropout, training=self.training)
#         return self.lin2(out)

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}('
#                 f'in_channels={self.in_channels}, '
#                 f'hidden_channels={self.hidden_channels}, '
#                 f'out_channels={self.out_channels}, '
#                 f'edge_dim={self.edge_dim}, '
#                 f'num_layers={self.num_layers}, '
#                 f'num_timesteps={self.num_timesteps}'
#                 f')')
                    