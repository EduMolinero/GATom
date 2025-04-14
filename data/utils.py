import numpy as np
import networkx as nx

from pymatgen.core.periodic_table import Element


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
    # add "node" attributes to the line graph
    for node in line_graph.nodes():
        distance = graph.edges[node]['distance']
        offset = graph.edges[node]['offset']
        line_graph.nodes[node]['distance'] = distance
        line_graph.nodes[node]['offset'] = offset
    
    # add "edge" attributes to the line graph
    for node_1, node_2, key in line_graph.edges(keys=True):
        distance_1, distance_2 = line_graph.nodes[node_1]['offset'], line_graph.nodes[node_2]['offset']
        line_graph.edges[node_1, node_2, key]['angle'] = angle(distance_1, distance_2)

    return line_graph
    




