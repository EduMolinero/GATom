import numpy as np
import networkx as nx

from pymatgen.core.periodic_table import Element

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    """
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)
    

def get_atomic_name(atomic_number: int):
    return Element.from_Z(atomic_number).symbol
    
def construct_graph_features(graph, atomic_encoding, max_distance=10, step=0.1):
    ## To do: implement this in a more general way so it accepts any attr string.
    # node features: dim(n_nodes, dim_feature)
    node_features = np.vstack(
        [atomic_encoding[get_atomic_name(attr['atomic_number'])] for nodes, attr in graph.nodes(data=True)]
    )

    filter = GaussianDistance(0, max_distance, step)
    # edge features: dim(n_edges, dim_features_edge)
    edge_features = np.vstack(
        [filter.expand(attr['distance']) for node1, node2, attr in graph.edges(data=True)]
    )

    return node_features, edge_features


def angle(vec1, vec2):
    """
    Calculate the x = cos(θ) between two vectors in radians.
    It is more numerically stable than computing θ = arccos(x).
    The angle is in the range [-1, 1].
    """
    x = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    x = np.clip(x, -1, 1)
    # Avoid numerical errors that cause x to be slightly greater than 1 or less than -1
    # due to floating point precision issues. 
    return x


def construct_line_graph(graph):
    """
    Create a line graph from the input graph.
    The line graph is a graph that represents the edges of the original graph as nodes.
    The edges of the line graph correspond to the edges of the original graph that share a common vertex.
    """
    line_graph = nx.line_graph(graph)
    for node in line_graph.nodes():
        offset = graph.edges[node]['offset']
        line_graph.nodes[node]['offset'] = offset    
    # add "edge" attributes to the line graph
    for node_1, node_2, key in line_graph.edges(keys=True):
        distance_1, distance_2 = line_graph.nodes[node_1]['offset'], line_graph.nodes[node_2]['offset']
        line_graph.edges[node_1, node_2, key]['angle'] = angle(distance_1, distance_2)

    return line_graph
    

def construct_line_graph_features(graph, step=0.1):
    #line graph
    line_graph = construct_line_graph(graph)
    # expand it into a Gaussian basis
    filter_angle = GaussianDistance(-1, 1, step)
    # get the actual features
    line_edge_features = np.vstack(
        [filter_angle.expand(attr['angle']) for node1, node2, attr in line_graph.edges(data=True)]
    )
    return line_edge_features



class GraphWithLineGraph(Data):
    """
    Class to store the graph and its line graph as a modified PyTorch Geometric Data class.
    It contains the following attributes:
        - x: node features
        - edge_index: edge indices
        - edge_attr: edge features
        - x_line: line graph node features
        - edge_index_line_graph: line graph edge indices
        - edge_attr_line_graph: line graph edge features
        - global_features: global features of the graph
        - y: target variable
    """
    def __inc__(self, key, value, *args, **kwargs):
        # self.x_line.size(0) is the number of edges in the original graph which now are the number of nodes
        # it would be equal to do edge_index.size(1)  
        if key == 'edge_index':
            return self.x.size(0) 
        if key == 'edge_index_line_graph':
            return self.x_line.size(0)
        return super().__inc__(key, value, *args, **kwargs)
    
    @classmethod
    def from_graph(cls, graph, atomic_encoding, global_features, y, max_distance=10, step=0.1):
        node_features, edge_features = construct_graph_features(graph, atomic_encoding, max_distance=max_distance, step=step)

        temp_data = Data(
                    x = torch.tensor(node_features, dtype=torch.float32),
                    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous(),
                    edge_attr = torch.tensor(edge_features, dtype=torch.float32),
        )

        # line graph
        line_graph = LineGraph(force_directed=True)(temp_data)
        edge_features_line = construct_line_graph_features(graph, step=step)

        data = cls(x=temp_data.x,
                   edge_index=temp_data.edge_index,
                   edge_attr=temp_data.edge_attr,
                   x_line=line_graph.x,
                   edge_index_line_graph=line_graph.edge_index,
                   edge_attr_line_graph=torch.tensor(edge_features_line, dtype=torch.float32),
                   global_features=torch.tensor(global_features, dtype=torch.float32),
                   y=torch.tensor([y], dtype=torch.float32),
                )

        return data




