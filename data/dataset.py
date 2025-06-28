import json
import os
import tempfile
import shutil
import multiprocessing as mp
from typing import List
from functools import partial

from tqdm import tqdm

from pymatgen.core import Structure
from pymatgen.symmetry.groups import SpaceGroup

import torch
from torch.nn.functional import one_hot

from torch_geometric.data import InMemoryDataset

import crystal_builder.prepocessor as cb

from .graphs import *


def load_atom_encoding(path):
    with open(path, 'r') as f:
        atom_encoding = json.load(f)
    return atom_encoding

def get_symmetry_group_onehot_dict() -> dict:
    """
    Generate a dictionary mapping space group numbers to one-hot encoded vectors.
    The number of crystallographic space groups is 230 in 3 dimensions.
    """
    num_groups = 230
    group_dict = {}
    for sg_num in range(1, num_groups + 1):
        # it is better to use only the int number of the space group rather than the symbol to avoid parsing errors.
        # sg = SpaceGroup.from_int_number(sg_num)
        # key = sg.symbol
        index = torch.tensor(sg_num - 1)
        one_hot_vector = one_hot(index, num_classes=num_groups)
        group_dict[sg_num] = one_hot_vector.tolist()
    return group_dict

def get_space_group_number(structure: Structure) -> int:
    """
    Get the integer that defines the space group of a structure.
    """
    symbol_sg, num_sg = structure.get_space_group_info()
    return num_sg

    
def bytes_to(bytes, to, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    return bytes / (bsize ** a[to])

def memory_estimation(data):
    """
    Estimation of the memory usage of a PyTorch Geometric Data object.
    """
    def tensor_memory(tensor):
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.nelement()

    total_memory = 0
    for key, value in data:
        if isinstance(value, torch.Tensor):
            total_memory += tensor_memory(value)
    return total_memory

def process_entry(entry, g, atomic_encoding, symmetry_group_encoding, line_graph_bool):
    structure = Structure.from_dict(entry[0])
    y = entry[1]
    try:
        graph = g(structure.remove_site_property('magmom'))
    except:
        graph = g(structure)

    data = GraphWithLineGraph.from_graph(
        graph=graph,
        atomic_encoding=atomic_encoding,
        global_features=symmetry_group_encoding[get_space_group_number(structure)],
        y=y,
        line_graph_bool=line_graph_bool,
        max_distance=8,
        step=0.2
    )
    # Estimation of memory usage
    memory_used = memory_estimation(data)
    
    return data, memory_used

def worker_func(entry):
    return process_entry(entry, g, atomic_encoding, symmetry_group_encoding, line_graph_bool)

def init_worker(_graph_algorithm, _atomic_enc, _symm_enc, _line_graph):
    """
    Per‑process set‑up.  The two big dicts come in as Manager‑proxies so every
    worker shares one physical copy; the graph‑builder is cheap, we create it
    once inside each child.
    """
    global g, atomic_encoding, symmetry_group_encoding, line_graph_bool
    g = getattr(cb, _graph_algorithm)()    
    atomic_encoding = _atomic_enc    
    symmetry_group_encoding = _symm_enc
    line_graph_bool = _line_graph


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
                line_graph_bool = True,
                graph_algorithm : str ='KNN',
                cell_type : str  = 'UnitCell',
        ):
        
        self._dataset_name = dataset_name
        self.num_samples = num_samples
        self._graph_algorithm = graph_algorithm + cell_type
        self._line_graph_bool = line_graph_bool
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [f'matbench_{self._dataset_name}.json']
        
    @property
    def processed_file_names(self) -> List[str]:
        return [f'matbench_{self._dataset_name}_{self._graph_algorithm}_LineG_{self._line_graph_bool}.pt']
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    
    def process(
        self,
        multiprocessing: bool = True,
        n_workers: int = 8,
        chunk_size: int = 1000,
    ) -> None:
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'r') as f:
                data_json = json.load(f)
            data_list = []
            total_memory = 0
            print(f'Staring processing data {raw_path}...')
            self.num_samples = (
                self.num_samples if self.num_samples is not None else len(data_json['data'])
            )
            entries = data_json['data'][: self.num_samples]
            del data_json

            atomic_encoding = load_atom_encoding(
                os.path.join(self.processed_dir, f"{self._dataset_name}_atom_features.json")
            )
            symmetry_group_encoding = get_symmetry_group_onehot_dict()
            
            tmp_dir = tempfile.mkdtemp(dir=self.processed_dir,
                                        prefix=f"_tmp_{self._dataset_name}_")

            data_list, chunk_paths, total_memory = [], [], 0

            # Process each entry in the dataset
            if multiprocessing:
                manager = mp.Manager()
                atomic_enc_share = manager.dict(atomic_encoding)
                symm_enc_share = manager.dict(symmetry_group_encoding)

                pool = mp.Pool(
                    processes=n_workers,
                    initializer=init_worker,
                    initargs=(self._graph_algorithm, atomic_enc_share, symm_enc_share, self._line_graph_bool),
                )

                try:
                    for idx, (data_obj, mem) in enumerate(
                            tqdm(
                                pool.imap(worker_func, entries, chunksize=16),
                                total=len(entries)
                            )
                        ):

                        data_list.append(data_obj)       
                        total_memory += mem

                        if len(data_list) >= chunk_size:
                            chunk_file = os.path.join(tmp_dir, f"{len(chunk_paths):08d}.pt")
                            torch.save(data_list, chunk_file)
                            chunk_paths.append(chunk_file)
                            data_list = []      

                finally:
                    pool.close()
                    manager.shutdown()
                    pool.join()

                if data_list:
                    chk_path = os.path.join(tmp_dir, f"{len(chunk_paths):08d}.pt")
                    torch.save(self.collate(data_list), chk_path)
                    chunk_paths.append(chk_path)

            else:   
                #create the graph builder object
                try:
                    g = getattr(cb, self._graph_algorithm)()
                    print(f'Using graph algorithm {self._graph_algorithm}')
                except:
                    raise ValueError(f'Graph algorithm {self._graph_algorithm} not implemented') 
                
                for idx, entry in enumerate(tqdm(entries)):
                    data_obj, mem = process_entry(entry, g, atomic_encoding, symmetry_group_encoding, self._line_graph_bool)
                    data_list.append(data_obj)
                    total_memory += mem

                    # flush to disk every <chunk_size> graphs
                    if len(data_list) >= chunk_size:
                        chunk_file = os.path.join(tmp_dir, f"{len(chunk_paths):08d}.pt")
                        torch.save(data_list, chunk_file)
                        chunk_paths.append(chunk_file)
                        data_list = []                 

                # flush the final partial chunk
                if data_list:
                    chunk_file = os.path.join(tmp_dir, f"{len(chunk_paths):08d}.pt")
                    torch.save(data_list, chunk_file)
                    chunk_paths.append(chunk_file)

            big_list = []
            for fp in sorted(chunk_paths):
                big_list.extend(torch.load(fp, weights_only=False))

            shutil.rmtree(tmp_dir, ignore_errors=True)       # tidy up
            torch.save(self.collate(big_list), processed_path)            

            print(f'Finished processing data {raw_path}...')
            print(f'Total estimated memory of the dataset: {bytes_to(total_memory, 'g')} GB')