import torch
import torch.nn.functional as F
from torch import Tensor


from torch_geometric.nn import GATv2Conv, DiffGroupNorm
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
    r"""

    Args:
    

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
        activation_cell: str = 'elu',
        pooling: str = 'global_add_pool',
        aggregation: str = 'add',
        activation: str = 'silu',
        heads: int = 1,
        residual_connection: bool = False,
        glu_over_gru: bool = False,
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
        self.glu_over_gru = glu_over_gru
        self.task = task
        self.pooling = pooling
        self.activation = activation 
        self.recurrent_cell_class = recurrent_cell_classes.get(recurrent_cell.lower())
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

        # Local attention
        self.embedding_att_convs = torch.nn.ModuleList()
        self.embedding_gate = torch.nn.ModuleList()
        self.embedding_norms = torch.nn.ModuleList()
        for _ in range(self.layers_attention - 1):
            
            self.embedding_att_convs.append(GATv2Conv(hidden_channels, hidden_channels, heads = self.heads, dropout=dropout,
                                            add_self_loops=False, negative_slope=0.01, edge_dim=hidden_channels,
                                            aggr = self.aggregation))
            # The output of the GATv2Conv has dimensions (N, out_channels * heads)
            if self.glu_over_gru:
                # the input dimension will be dim(x) + dim(h) = hidden_channels +  hidden_channels * heads = hidden_channels * (heads + 1)
                self.embedding_gate.append(GeneralGLU(hidden_channels * (self.heads + 1), hidden_channels, activation = self.activation_cell))
            else:
                # We need to properly set up the dimensions of the h state in the GRUCell
                self.embedding_gate.append(self.recurrent_cell_class(hidden_channels * self.heads, hidden_channels))
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
        if self.glu_over_gru:
            self.global_gate = GeneralGLU(2 * hidden_channels, hidden_channels, activation = self.activation_cell)
        else:
            self.global_gate = self.recurrent_cell_class(hidden_channels, hidden_channels)
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

        for att_conv, gru, norm in zip(self.embedding_att_convs, self.embedding_gate, self.embedding_norms):
            att_conv.reset_parameters()
            gru.reset_parameters()
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
            # Conv :  GAT + GRU -> g_l := Conv(x_l)
            # Res+ : x_l = g_{l-1} + x_{l-1}
            if self.res_connection:
                g = norm(x)
                g = F.relu(g)
                g = F.dropout(g, p=self.dropout, training=self.training)
                if self.glu_over_gru:
                    h = att_conv(g, edge_index, edge_attr)
                    x = gate(torch.cat([x, h], dim=-1))
                else:
                    for t in range(self.num_timesteps):
                        h = getattr(F, self.activation_cell)(att_conv(g, edge_index, edge_attr))
                        g = gate(h, g).relu()
                x = x + g
                ## Res connection
                # Conv -> Norm -> activation -> Res
                # for t in range(self.num_timesteps):
                #     h = getattr(F, self.activation_cell)(att_conv(x, edge_index, edge_attr))
                #     h = F.dropout(h, p=self.dropout, training=self.training)
                #     g = gru(h, x).relu()
                # g = norm(g)
                # x = x + F.relu(g)
            else:
                if self.glu_over_gru:
                    h = att_conv(x, edge_index, edge_attr)
                    x = gate(torch.cat([x, h], dim=-1))
                else:
                    for t in range(self.num_timesteps):
                        h = getattr(F, self.activation_cell)(att_conv(x, edge_index, edge_attr))
                        h = F.dropout(h, p=self.dropout, training=self.training)
                        x = gate(h, x).relu()
                x = norm(x)

        # Global attention
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = getattr(torch_geometric.nn, self.pooling)(x, batch).relu()
        if self.glu_over_gru: 
            h = self.global_conv((x, out), edge_index)
            out = self.global_gate(torch.cat([out, h], dim=-1))
        else:
            for t in range(self.num_timesteps):
                h = getattr(F, self.activation_cell)(self.global_conv((x, out), edge_index))
                h = F.dropout(h, p=self.dropout, training=self.training)
                out = self.global_gate(h, out).relu()

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
                f'num_timesteps={self.num_timesteps},'
                f'aggregation={self.aggregation},'
                f'heads={self.heads},'
                f'residual_connection={self.res_connection},'
                f'dropout={self.dropout:.2f},'
                f'task={self.task},'
                f'pooling={self.pooling},'
                f'activation={self.activation},'
                f'recurrent_cell={self.recurrent_cell_class.__name__}'
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
            f"    glu_over_gru         = {self.glu_over_gru},\n"
            f"    gate                 = {f"GeneralGLU" if self.glu_over_gru else self.recurrent_cell_class.__name__}\n"
            f"    activation_cell      = {self.activation_cell}\n"
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
            "glu_over_gru" : self.glu_over_gru,
            "recurrent_cell" : "GeneralGLU" if self.glu_over_gru else self.recurrent_cell_class.__name__,
            "activation_cell" : self.activation_cell
        }