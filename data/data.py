import json
import os
from typing import List

import numpy as np
from tqdm import tqdm

from pymatgen.core import Structure
from pymatgen.symmetry.groups import SpaceGroup

import torch
from torch.nn.functional import one_hot

from torch_geometric.data import Data, InMemoryDataset

import crystal_builder.prepocessor as cb
from utils import *


def load_atom_encoding(path):
    with open(path, 'r') as f:
        atom_encoding = json.load(f)
    return atom_encoding

def get_symmetry_group_onehot_dict():
    """
    Generate a dictionary mapping space group numbers to one-hot encoded vectors.
    The number of crystallographic space groups is 230 in 3 dimensions.
    """
    num_groups = 230
    group_dict = {}
    for sg_num in range(1, num_groups + 1):
        sg = SpaceGroup.from_int_number(sg_num)
        key = sg.symbol
        index = torch.tensor(sg_num - 1)
        one_hot_vector = one_hot(index, num_classes=num_groups)
        group_dict[key] = one_hot_vector.tolist()
    return group_dict

# Generate the dictionary
symmetry_group_dict = get_symmetry_group_onehot_dict()

# For demonstration, print out the key and one-hot vector of a couple of groups.
for key in list(symmetry_group_dict.keys())[:3]:
    print(f"{key}: {symmetry_group_dict[key]}")

    
def bytes_to(bytes, to, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    return bytes / (bsize ** a[to])


class MatbenchDataset(InMemoryDataset):
    """
    Matbench dataset class for PyTorch
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
            print(f'Staring processing data {raw_path}...')
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

            # load symmetry group encoding
            symmetry_group_encoding = get_symmetry_group_onehot_dict()
            
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

                # features from original graph
                node_features, edge_features = construct_graph_features(graph, atomic_enconding)
                global_features = symmetry_group_encoding[structure.get_space_group_info()[0]]

                # create line graph
                line_graph = construct_line_graph(graph)
                #line_node_features, line_edge_features = construct_line_graph_features(line_graph, edge_features)

                data = Data(
                    x = torch.tensor(node_features, dtype=torch.float32),
                    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous(),
                    edge_attr = torch.tensor(edge_features, dtype=torch.float32),
                    y = torch.tensor([y], dtype=torch.float32)
                )

                data_list.append(data)
                total_memory += data.num_edges * 2 * data.edge_attr.element_size() + data.num_nodes * data.x.element_size()
                print(f'Memory used: {bytes_to(total_memory, "m"):.4f} MB')

            
            torch.save(self.collate(data_list), processed_path)
            print(f'Finished processing data {raw_path}...')