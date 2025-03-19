import numpy as np
import json
import re
import os

from typing import Callable, List, Optional, Union, Dict

from scipy.ndimage import gaussian_filter1d

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from pymatgen.analysis.local_env import NearNeighbors, VoronoiNN, IsayevNN, CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_networkx, add_self_loops
import torch_geometric.transforms as T
import torch_geometric.utils
import torch.nn.functional as F


import crystal_builder.prepocessor as cb

    
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
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

def load_atom_encoding(path):
    with open(path, 'r') as f:
        atom_encoding = json.load(f)
    return atom_encoding
    
def bytes_to(bytes, to, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    return bytes / (bsize ** a[to])

def get_atomic_name(atomic_number: int):
    return Element.from_Z(atomic_number).symbol


def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = torch_geometric.utils.degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


class MatbenchDataset(InMemoryDataset):
    """
    Matbench dataset class for PyTorch
    Very (really bad) poorly implementented --> Improve it
    To do: implenten things as download_url, etc.
    """
    def __init__(self, 
                root,
                dataset_name='mp_gap',
                transform=None,
                pre_transform=None,
                pre_filter=None,
                num_samples=None, 
                force_reload=False,
                graph_algorithm : str ='KNN', 
                cell_type : str  = 'UnitCell',
                ):
        
        self._dataset_name = dataset_name
        self.num_samples = num_samples
        self._graph_algorithm = graph_algorithm + cell_type
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [f'matbench_{self._dataset_name}.json']
        
    @property
    def processed_file_names(self) -> List[str]:
        return [f'matbench_{self._dataset_name}_{self._graph_algorithm}.pt']
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    
    def process(self) -> None:
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'r') as f:
                data = json.load(f)
            data_list = []
            total_memory = 0
            print('Staring processing data {}...'.format(raw_path))
            self.num_samples = self.num_samples if self.num_samples is not None else len(data['data'])

            #create the graph builder object
            try:
                g = getattr(cb, self._graph_algorithm)()
                print(f'Using graph algorithm {self._graph_algorithm}')
            except:
                raise ValueError(f'Graph algorithm {self._graph_algorithm} not implemented')
            
            # load atomic encoding
            atomic_enconding = load_atom_encoding(
                os.path.join(self.processed_dir, f'{self._dataset_name}_atom_features.json')
            )

            # set up class for gaussian expansion
            filter = GaussianDistance(0,8, 0.2)
            
            for i, entry in enumerate(tqdm(data['data'])):
                if i >= self.num_samples:
                    break
                structure = Structure.from_dict(entry[0])
                y = entry[1]
                # we remove the magnetic orbital moment since it is something we don't use
                try:
                    graph = g(structure.remove_site_property('magmom'))
                except: 
                    graph = g(structure)
                
                ## To do: implement this in a more general way so it accepts any attr string. 
                # node features: dim(n_nodes, dim_feature)
                node_features = np.vstack(
                    [atomic_enconding[get_atomic_name(attr['atomic_number'])] for nodes, attr in graph.nodes(data=True)]
                )

                # edge features: dim(n_edges, dim_features_edge)
                edge_features = np.vstack(
                    [filter.expand(attr['distance']) for node1, node2, attr in graph.edges(data=True)]
                )

                data = Data(
                    x = torch.tensor(node_features, dtype=torch.float32),
                    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous(),
                    edge_attr = torch.tensor(edge_features, dtype=torch.float32),
                    y = torch.tensor([y], dtype=torch.float32)
                )

                data_list.append(data)
                total_memory += data.num_edges * 2 * data.edge_attr.element_size() + data.num_nodes * data.x.element_size()
                print(f'Memory used: {bytes_to(total_memory, "m"):.4f} MB')

            #Now we perform several operations on the dataset
            # for index in range(0, len(data_list)):
            #     # we add the degree of each node as a one hot encoding
            #     data_list[index] = OneHotDegree(
            #         data_list[index], max_degree=12, in_degree=False, cat=True
            #     )
            
            torch.save(self.collate(data_list), processed_path)