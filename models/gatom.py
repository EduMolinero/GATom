import torch
import torch.nn.functional as F   
from torch import Tensor

import torch_geometric
from torch_geometric.typing import Tensor, OptTensor
from torch_geometric.nn import aggr, GATv2Conv, LayerNorm, TransformerConv, DiffGroupNorm
from torch_geometric.nn.dense.linear import Linear

from data.graphs import GraphWithLineGraph

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

class LocalAttentionBlock(torch.nn.Module):
    """
    Local Attention Block that applies a GATv2 -> GLU - > LayerNorm -> Residual connection.
    It is valid whether to process the original graph or a line graph.
    Args:
        node_dim (int): 
            Dimensionality of the node embeddings.
        edge_dim (int): 
            Number of input features per edge.
        aggregation (str): 
            Aggregation strategy for GATv2Conv (e.g., "add", "softmax", "powermean"). 
        activation (str): 
            Activation function for GLU. 
        heads (int): 
            Number of attention heads for local attention. 
        dropout (float): 
            Dropout probability for attention layers. 
        concat (bool): 
            Whether to concatenate the x and the output of the GATv2Conv layer. If False it only uses the results of the GATv2Conv layer.
    """
    def __init__(
            self, 
            node_dim, 
            edge_dim, 
            aggregation='add', 
            activation = 'relu', 
            heads=4, 
            dropout=0.0, 
        ):
        super().__init__()
        self.activation = activation
        self.dropout = dropout
        # GATv2Conv layer
        self.norm = LayerNorm(node_dim)
        self.att_conv = GATv2Conv(
            node_dim, 
            node_dim, 
            heads=heads, 
            dropout=dropout,
            add_self_loops=True,
            negative_slope=0.01,
            edge_dim=edge_dim,
            aggr=aggregation
        )
        self.gate = GeneralGLU(node_dim * (heads + 1), node_dim, activation=activation)

        # # GATv2Conv layer
        # self.norm_att = LayerNorm(node_dim)
        # self.att_conv = GATv2Conv(
        #     node_dim, 
        #     node_dim, 
        #     heads=heads, 
        #     dropout=dropout,
        #     add_self_loops=True,
        #     negative_slope=0.01,
        #     edge_dim=edge_dim,
        #     aggr=aggregation
        # )
        # # The output of the GATv2Conv has dimensions (N, out_channels * heads)
        # # We mix the heads together
        # self.linear_heads = Linear(node_dim * heads, node_dim, bias=False)

        # # GLU and FFN layer
        # self.norm_ffn = LayerNorm(node_dim)
        # self.gate = GeneralGLU(node_dim, node_dim, activation=activation)
        # self.ffn = Linear(node_dim, node_dim, bias=False)
        # # FFN Normalization layer

    def forward(self, x, edge_index, edge_attr):
        # # Apply Norm & GATv2Conv
        # g = self.att_conv(self.norm_att(x), edge_index, edge_attr)
        # # Residual connection
        # x = x + self.linear_heads(g)

        # # Apply Norm & GLU and FFN
        # X = self.gate(self.norm_ffn(x))
        # # FFN & Residual connection
        # x = x + self.ffn(X)
        ## Res+ strategy
        # Norm 
        x = self.norm(x)
        # Activation
        x = getattr(F, self.activation)(x)
        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # GATv2Conv & Gate
        g = self.att_conv(x, edge_index, edge_attr)
        X = self.gate(torch.cat([x, g], dim=-1))
        # Residual Connection
        x = x + X
        return x    

class LocalAttentionLayer(torch.nn.Module):
    """
    This class implementes the Local Attention Block as a stack of two LocalAttentionBlocks.
    The first block deals with the line graph and the second block deals with the original graph.
    """
    def __init__(
            self, 
            node_dim, 
            edge_dim,
            line_edge_dim,
            aggregation='add', 
            activation = 'silu',
            heads=4,
            dropout=0.0,
        ):
        super().__init__()

        # Process the line graph
        # Edge_dim is equal to the node_dim of the line graph
        self.line_graph_block = LocalAttentionBlock(
            edge_dim,
            line_edge_dim,
            aggregation=aggregation,
            activation=activation,
            heads=heads,
            dropout=dropout,
        )
        # Process the original graph
        self.graph_block = LocalAttentionBlock(
            node_dim,
            edge_dim,
            aggregation=aggregation,
            activation=activation,
            heads=heads,
            dropout=dropout,
        )

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            x_line: Tensor,
            edge_index_line: Tensor,
            edge_attr_line: Tensor,
        ):

        # Apply the first block to the line graph
        x_line = self.line_graph_block(x_line, edge_index_line, edge_attr_line)
        # Apply the second block to the original graph
        # x_line == edge_attr
        x = self.graph_block(x, edge_index, x_line)
        return x, x_line


class GlobalPooling(torch.nn.Module):
    """
    Global pooling of the node features into a vector.
    It combines the pooling with the global features of each graph.
    """
    def __init__(
            self, 
            node_dim, 
            global_fea_dim,
            pooling='global_mean_pool',
            activation='silu',
            concat=True,
        ):
        super().__init__()
        self.pooling = pooling
        self.activation = activation
        self.global_fea_dim = global_fea_dim
        self.concat = concat
        self.dim_gate = 3 * node_dim if concat else node_dim

        # We match the global feature dimension to the node dimension
        self.global_lin = Linear(global_fea_dim, node_dim, bias=False)
        # final layer 
        self.global_gate = GeneralGLU(self.dim_gate, node_dim, activation=activation)


    def forward(self, x: torch.Tensor, x_line: torch.Tensor, batch: torch.Tensor, batch_line: torch.Tensor, g_feat: torch.Tensor): 

        # Pool node features
        pooled = getattr(torch_geometric.nn, self.pooling)(x, batch)
        # Pool line graph features
        pooled_line = getattr(torch_geometric.nn, self.pooling)(x_line, batch_line)
        #reshape the tensor into [batch_size, global_fea_dim]
        g_feat = g_feat.view(-1, self.global_fea_dim)
        g_emb  = self.global_lin(g_feat) 

        # Combine everything together
        super_x = torch.cat((pooled, pooled_line, g_emb), dim=-1) if self.concat else pooled + g_emb + pooled_line 
        super_x = self.global_gate(super_x)

        return super_x


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
        task (str, optional): 
            Model task type: "classification" or "regression". Defaults to "regression".
        dropout (float, optional): 
            Dropout probability for attention layers. Defaults to 0.0.
        pre_conv_layers (int, optional): 
            Number of linear layers to transform node/edge features before GAT. Defaults to 1.
        post_conv_layers (int, optional): 
            Number of linear layers to transform the final node embedding before prediction. Defaults to 2.

    Returns:
        torch.Tensor: Model output of shape (batch_size, out_channels), or a flattened shape (batch_size,) for a single output dimension.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        line_edge_dim: int,
        global_fea_dim: int,
        pre_conv_layers: int = 1,
        layers_attention: int = 5,
        layers_attention_line: int = 2,
        post_conv_layers: int = 2,
        activation_cell: str = 'elu',
        pooling: str = 'global_add_pool',
        aggregation: str = 'add',
        activation: str = 'silu',
        heads: int = 1,
        task: str = 'regression',
        dropout: float = 0.0,
        line_graph: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.line_edge_dim = line_edge_dim
        self.global_fea_dim = global_fea_dim
        self.layers_attention = layers_attention
        self.layers_attention_line = layers_attention_line
        self.dropout = dropout
        self.heads = heads
        self.task = task
        self.pooling = pooling
        self.activation = activation
        self.activation_cell = activation_cell
        self.pre_conv_layers = pre_conv_layers
        self.post_conv_layers = post_conv_layers
        self.line_graph = line_graph


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
        self.pre_conv_edges_line = torch.nn.ModuleList()
        for i in range(self.pre_conv_layers):
            if i == 0:
                self.pre_conv_nodes.append(Linear(in_channels, hidden_channels))
                self.pre_conv_edges.append(Linear(edge_dim, hidden_channels))
                if self.line_graph:
                    self.pre_conv_edges_line.append(Linear(line_edge_dim, hidden_channels))
            else:
                self.pre_conv_nodes.append(Linear(hidden_channels, hidden_channels))
                self.pre_conv_edges.append(Linear(hidden_channels, hidden_channels))
                if self.line_graph:
                    self.pre_conv_edges_line.append(Linear(hidden_channels, hidden_channels))

        #################################################################
        # Local attention
        self.local_attention_line = torch.nn.ModuleList()
        self.local_attention = torch.nn.ModuleList()
        if self.line_graph:
            for _ in range(self.layers_attention_line - 1):
                self.local_attention_line.append(LocalAttentionBlock(
                    node_dim=hidden_channels,
                    edge_dim=hidden_channels,
                    aggregation=self.aggregation,
                    activation=self.activation_cell,
                    heads=self.heads,
                    dropout=self.dropout,
                ))
            for _ in range(self.layers_attention - 1):
                self.local_attention.append(LocalAttentionBlock(
                    node_dim=hidden_channels,
                    edge_dim=hidden_channels,
                    aggregation=self.aggregation,
                    activation=self.activation_cell,
                    heads=self.heads,
                    dropout=self.dropout,
                ))

        else:
            for _ in range(self.layers_attention - 1):
                self.local_attention.append(LocalAttentionBlock(
                    node_dim=hidden_channels,
                    edge_dim=hidden_channels,
                    aggregation=self.aggregation,
                    activation=self.activation_cell,
                    heads=self.heads,
                    dropout=self.dropout,
                ))

        #################################################################
        # Global attention
        self.global_pooling = GlobalPooling(
            node_dim=hidden_channels,
            global_fea_dim=global_fea_dim,
            pooling=self.pooling,
            activation=self.activation,
        )

        #################################################################
        # Readout block
        self.post_conv_hidden = torch.nn.ModuleList()
        for _ in range(self.post_conv_layers):
            self.post_conv_hidden.append(Linear(hidden_channels, hidden_channels))

        self.output_layer = Linear(hidden_channels, out_channels)

    def forward(self, data: GraphWithLineGraph) -> Tensor:
        # split different features
        x, edge_index, batch  = data.x, data.edge_index, data.batch
        global_features = data.global_features

        if self.line_graph:
            x_line, edge_index_line, edge_attr_line, batch_line = data.x_line, data.edge_index_line, data.edge_attr_line, data.x_line_batch
        else:
            x_line, batch_line = data.x_line, data.x_line_batch

        #Embedding block
        if self.line_graph:
            for pre_node, pre_edge, pred_edge_line in zip(self.pre_conv_nodes, self.pre_conv_edges, self.pre_conv_edges_line):
                x = getattr(F, self.activation)(pre_node(x))
                x_line = getattr(F, self.activation)(pre_edge(x_line))
                edge_attr_line = getattr(F, self.activation)(pred_edge_line(edge_attr_line))
        else:
            for pre_node, pre_edge in zip(self.pre_conv_nodes, self.pre_conv_edges):
                x = getattr(F, self.activation)(pre_node(x))
                x_line = getattr(F, self.activation)(pre_edge(x_line))

        # Local attention
        if self.line_graph:
            for local_attention_layer in self.local_attention_line:
                # x, x_line = local_attention_layer(
                #     x, edge_index, x_line, edge_index_line, edge_attr_line
                # )
                x_line = local_attention_layer(x_line,edge_index_line, edge_attr_line)
            for local_attention_layer in self.local_attention:
                # x, x_line = local_attention_layer(
                #     x, edge_index, x_line, edge_index_line, edge_attr_line
                # )
                x = local_attention_layer(x, edge_index, x_line)
            
                # apply dropout
                #x = F.dropout(x, p=self.dropout, training=self.training)
                #x_line = F.dropout(x_line, p=self.dropout, training=self.training)
        else:
            for local_attention_layer in self.local_attention:
                x = local_attention_layer(x, edge_index, x_line)
                # apply dropout
                #x = F.dropout(x, p=self.dropout, training=self.training)

        # Global attention: out is the supernode
        out = self.global_pooling(x, x_line, batch, batch_line, global_features)
        out = F.dropout(out, p=self.dropout, training=self.training)

        # Readout block
        for hidden_layer in self.post_conv_hidden:
            out = getattr(F, self.activation)(hidden_layer(out))

        # Predictor:
        if self.task == 'classification':
            out = F.sigmoid(self.output_layer(out))
        elif self.task == 'regression':
            out = self.output_layer(out)
        else:
            raise ValueError('Task not recognized.')
    
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'line_edge_dim={self.line_edge_dim}, '
                f'global_fea_dim={self.global_fea_dim}, '
                f'layers_attention={self.layers_attention},'
                f'layers_attention_line={self.layers_attention_line if self.line_graph else None},'
                f'pre_conv_layers={self.pre_conv_layers},'
                f'post_conv_layers={self.post_conv_layers},'
                f'aggregation={self.aggregation},'
                f'heads={self.heads},'
                f'dropout={self.dropout:.2f},'
                f'task={self.task},'
                f'pooling={self.pooling},'
                f'activation={self.activation},'
                f'activation_cell={self.activation_cell}'
                f'line_graph={self.line_graph}'
                f')')
    
    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"    in_channels          = {self.in_channels},\n"
            f"    hidden_channels      = {self.hidden_channels},\n"
            f"    out_channels         = {self.out_channels},\n"
            f"    edge_dim             = {self.edge_dim},\n"
            f"    line_edge_dim        = {self.line_edge_dim},\n"
            f"    global_fea_dim       = {self.global_fea_dim},\n"
            f"    pre_conv_layers      = {self.pre_conv_layers},\n"
            f"    layers_attention     = {self.layers_attention},\n"
            f"    layers_att_line      = {self.layers_attention_line if self.line_graph else None},\n"
            f"    post_conv_layers     = {self.post_conv_layers},\n"
            f"    aggregation          = {self.aggregation},\n"
            f"    heads                = {self.heads},\n"
            f"    dropout              = {self.dropout:.2f},\n"
            f"    task                 = {self.task},\n"
            f"    pooling              = {self.pooling},\n"
            f"    activation           = {self.activation},\n"
            f"    gate                 = {f"GLU with {self.activation_cell}"}\n"
            f"    line_graph           = {self.line_graph}\n"
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
            "line_edge_dim" : self.line_edge_dim,
            "global_fea_dim" : self.global_fea_dim,
            "layers_attention" : self.layers_attention,
            "pre_conv_layers" : self.pre_conv_layers,
            "post_conv_layers" : self.post_conv_layers,
            "aggregation" : self.aggregation,
            "heads" : self.heads,
            "dropout" : self.dropout,
            "task" : self.task,
            "pooling" : self.pooling,
            "activation" : self.activation,
            "activation_cell" : self.activation_cell,
            "line_graph": self.line_graph
        }