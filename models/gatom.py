import torch
import torch.nn.functional as F
from torch import Tensor


from torch_geometric.nn import GATv2Conv, DiffGroupNorm, LayerNorm
from torch_geometric.nn.dense.linear import Linear
                    
from torch import Tensor
from torch.nn import GRUCell, Linear
from .mgu import liGRUCell, MGUCell

import torch_geometric
from torch_geometric.typing import Tensor, OptTensor
from torch_geometric.nn import aggr


recurrent_cell_classes = {
    'gru': GRUCell,
    'ligru': liGRUCell,
    'mgu' : MGUCell,    
}


class GeneralGLU(torch.nn.Module):
    """
    Custom implmentation of a Gated Linear Unit (GLU) with a gating mechanism.
    It can use any activation function, so we use different gates as GLU, GEGLU, etc.
    """
    def __init__(
            self, 
            input_dim, 
            output_dim,
            activation,
        ):
        super().__init__()
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        # The linear layer produces 2 * output_dim:
        #  - the first half will be the "values" part
        #  - the second half will be the "gate" part
        self.linear = torch.nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x):
        # x: (..., input_dim)
        # Project x into two parts
        projected = self.linear(x)  # (..., 2 * output_dim)
        values, gates = projected.chunk(2, dim=-1)
        return values * getattr(F, self.activation)(gates)

    def reset_parameters(self, gate_bias=1.0):
        # Initialize the linear layer with xavier uniform
        torch.nn.init.xavier_uniform_(self.linear.weight)

        # Initialize all biases to 0
        torch.nn.init.zeros_(self.linear.bias)

        # Then set the second half (the gate portion) to gate_bias
        with torch.no_grad():
            self.linear.bias[self.output_dim:] = gate_bias

                    
class GATom(torch.nn.Module):
    """
    A Graph Attention Network (GAT) model that integrates both local and global
    attention mechanisms, using GLU gating, and supports
    residual connections for graph-level classification or regression tasks.

    Args:
        in_channels (int): 
            Number of input features per node.
        hidden_channels (int): 
            Dimensionality of the hidden node embeddings.
        out_channels (int): 
            Number of output features for the final prediction.
        edge_dim (int): 
            Number of input features per edge.
        layers_attention (int): 
            Number of local attention layers to stack. Defaults to 4.
        activation_cell (str, optional): 
            Activation to use inside the gating cell (e.g., "relu", "elu"). Defaults to "elu".
        pooling (str, optional): 
            Graph pooling method for readout (e.g., "global_add_pool"). Defaults to "global_add_pool".
        aggregation (str, optional): 
            Aggregation strategy for GATv2Conv (e.g., "add", "softmax", "powermean"). Defaults to "add".
        activation (str, optional): 
            Top-level activation for hidden layers (e.g., "silu", "relu"). Defaults to "silu".
        heads (int, optional): 
            Number of attention heads for local attention. Defaults to 1.
        residual_connection (bool, optional): 
            Whether to add skip connections between layers. Defaults to False.
        task (str, optional): 
            Model task type: "classification" or "regression". Defaults to "regression".
        dropout (float, optional): 
            Dropout probability for attention layers. Defaults to 0.0.
        pre_conv_layers (int, optional): 
            Number of linear layers to transform node/edge features before GAT. Defaults to 1.
        post_conv_layers (int, optional): 
            Number of linear layers to transform the final node embedding before prediction. Defaults to 2.

    Attributes:
        pre_conv_nodes (torch.nn.ModuleList): 
            Linear layers for node feature transformation prior to GAT.
        pre_conv_edges (torch.nn.ModuleList): 
            Linear layers for edge feature transformation prior to GAT.
        embedding_att_convs (torch.nn.ModuleList): 
            List of GATv2Conv layers for local attention.
        embedding_gate (torch.nn.ModuleList): 
            Gate layers (GRUCell or GLU) applied after local attention.
        embedding_norms (torch.nn.ModuleList): 
            Normalization layers (e.g., LayerNorm or DiffGroupNorm) for each local attention layer.
        global_conv (GATv2Conv): 
            Single-head GAT layer for global attention via a super-node approach.
        global_gate (torch.nn.Module): 
            Gate for global attention (GRUCell or GLU).
        global_norm (torch.nn.Module): 
            Normalization layer applied after global attention pooling.
        post_conv_hidden (torch.nn.ModuleList): 
            Linear layers after attention layers for further transformation.
        output_layer (torch.nn.Linear): 
            Final linear layer to generate classification or regression outputs.

    Returns:
        torch.Tensor: Model output of shape (batch_size, out_channels), or a flattened shape (batch_size,) for a single output dimension.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        pre_conv_layers: int = 1,
        layers_attention: int = 4,
        post_conv_layers: int = 2,
        activation_cell: str = 'elu',
        pooling: str = 'global_add_pool',
        aggregation: str = 'add',
        activation: str = 'silu',
        heads: int = 1,
        residual_connection: bool = False,
        task: str = 'regression',
        dropout: float = 0.0,

    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim 
        self.layers_attention = layers_attention
        self.dropout = dropout
        self.heads = heads
        self.res_connection = residual_connection
        self.task = task
        self.pooling = pooling
        self.activation = activation 
        self.activation_cell = activation_cell
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
        # Local attention
        self.embedding_att_convs = torch.nn.ModuleList()
        self.embedding_gate = torch.nn.ModuleList()
        self.embedding_norms = torch.nn.ModuleList()
        for _ in range(self.layers_attention - 1):

            #norm layer  
            #self.embedding_norms.append(DiffGroupNorm(hidden_channels, groups=10))
            self.embedding_norms.append(LayerNorm(hidden_channels))

            self.embedding_att_convs.append(GATv2Conv(hidden_channels, hidden_channels, heads = self.heads, dropout=dropout,
                                            add_self_loops=False, negative_slope=0.01, edge_dim=hidden_channels,
                                            aggr = self.aggregation))
            # The output of the GATv2Conv has dimensions (N, out_channels * heads)
            # the input dimension will be dim(x) + dim(h) = hidden_channels +  hidden_channels * heads = hidden_channels * (heads + 1)
            self.embedding_gate.append(GeneralGLU(hidden_channels * (self.heads + 1), hidden_channels, activation = self.activation_cell))
            
            
        #################################################################
        # Global attention
        # In order to have a global attention we need to add a super node to the graph
        # We add a super node to the graph with only one head of attention
        # otherwise we this leads to problems with the global pooling
        self.global_conv = GATv2Conv(hidden_channels, hidden_channels, heads = 1, dropout=dropout,
                                    add_self_loops=False, negative_slope=0.01, edge_dim=hidden_channels,
                                    aggr = self.aggregation)
        self.global_gate = GeneralGLU(2 * hidden_channels, hidden_channels, activation = self.activation_cell)
        #self.global_norm = DiffGroupNorm(hidden_channels, groups=10)
        self.global_norm = LayerNorm(hidden_channels)

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

        for att_conv, gate, norm in zip(self.embedding_att_convs, self.embedding_gate, self.embedding_norms):
            att_conv.reset_parameters()
            gate.reset_parameters()
            norm.reset_parameters()

        for hidden_layer in self.post_conv_hidden:
            hidden_layer.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: OptTensor,
                batch: Tensor) -> Tensor:
        
        #Embedding block
        for pre_node, pre_edge in zip(self.pre_conv_nodes, self.pre_conv_edges):
            x = getattr(F, self.activation)(pre_node(x))
            edge_attr = getattr(F, self.activation)(pre_edge(edge_attr))

        # attention mechanism
        # Local attention
        for index_layer, (att_conv, gate, norm) in enumerate(zip(self.embedding_att_convs, self.embedding_gate, self.embedding_norms)):
            # we follow the skip connection (Res+) strategy
            # Normalization -> Activation -> Dropout -> Conv -> Res
            # Conv :  GAT + GLU -> g_l := Conv(x_l)
            # Res+ : x_l = g_{l-1} + x_{l-1}
            if self.res_connection:
                x = norm(x)
                x = getattr(F, self.activation)(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                g = att_conv(x, edge_index, edge_attr)
                X = gate(torch.cat([x, g], dim=-1))
                x = x + X
            else:
                x = norm(x)
                g = att_conv(x, edge_index, edge_attr)
                x = gate(torch.cat([x, g], dim=-1))

        # Global attention
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = getattr(torch_geometric.nn, self.pooling)(x, batch).relu()
        g = self.global_conv((x, out), edge_index)
        out = self.global_gate(torch.cat([out, g], dim=-1))
        x = self.global_norm(out)

        # Readout block
        for hidden_layer in self.post_conv_hidden:
            x = getattr(F, self.activation)(hidden_layer(x))
        
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
                f'aggregation={self.aggregation},'
                f'heads={self.heads},'
                f'residual_connection={self.res_connection},'
                f'dropout={self.dropout:.2f},'
                f'task={self.task},'
                f'pooling={self.pooling},'
                f'activation={self.activation},'
                f'activation_cell={self.activation_cell}'
                f')')
    
    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"    in_channels          = {self.in_channels},\n"
            f"    hidden_channels      = {self.hidden_channels},\n"
            f"    out_channels         = {self.out_channels},\n"
            f"    edge_dim             = {self.edge_dim},\n"
            f"    pre_conv_layers      = {self.pre_conv_layers},\n"
            f"    layers_attention     = {self.layers_attention},\n"
            f"    post_conv_layers     = {self.post_conv_layers},\n"
            f"    aggregation          = {self.aggregation},\n"
            f"    heads                = {self.heads},\n"
            f"    residual_connection  = {self.res_connection},\n"
            f"    dropout              = {self.dropout:.2f},\n"
            f"    task                 = {self.task},\n"
            f"    pooling              = {self.pooling},\n"
            f"    activation           = {self.activation},\n"
            f"    gate                 = {f"GLU with {self.activation_cell}"}\n"
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
            "aggregation" : self.aggregation,
            "heads" : self.heads,
            "res_connection" : self.res_connection,
            "dropout" : self.dropout,
            "task" : self.task,
            "pooling" : self.pooling,
            "activation" : self.activation,
            "activation_cell" : self.activation_cell
        }