import json
from typing import List, Dict, AnyStr, Union

import numpy as np
from pymatgen.core import Structure
from tqdm import tqdm

from torch.nn.functional import one_hot
from torch import arange as torch_arange

class AtomLoader:
    """
    Class to load atom information from a dataset.
    """

    def __init__(self, unique_elements: List[AnyStr]):
        self._unique_elements = unique_elements 
        self._total_elements = len(unique_elements)
    
    @property
    def total_elements(self):
        return self._total_elements
    
    @property
    def unique_elements(self):
        return self._unique_elements
    
    @property
    def tableau_categories(self):
        """
        Return the categories of the vector representation of the elements according to PRL 120, 145301 (2018).
        For numerical categories (e.g. group) only the number of categories is stored.
        For physical quantities (e.g. electronegativity) the number of categories is stored as well as the minimum and maximum values in the following way:
            [number of categories, (min value, max value)]

        We have modified the original categories to include the f-like orbitals of the of the element. This meant that:
        valence: 12 -> 14
        """
        tableu = {
            "group" : 18,
            "period" : 9,
            "electronegativity" : [10, (0.5, 4.0)],
            "covalent_radius" : [10, (25, 250)],
            "valence" : 14,
            "ionization_energy" : [10, (1.3, 3.3)],
            "electron_affinity" : [10, (-3.0, 3.7)],
            "block" : 4, 
            "molar_volume" : [10, (1.5, 4.3)]
        }
        dim = sum([x if type(x) == int else x[0] for x in tableu.values()])
        return tableu, dim

    
    @classmethod
    def from_matbench_dataset(cls, dataset: List[Dict]):
        unique_elements = []
        for entry in tqdm(dataset):
            structure = Structure.from_dict(entry[0])
            atoms = np.unique([site.species_string for site in structure.sites])
            unique_elements = np.append(unique_elements, atoms)
            unique_elements = np.unique(unique_elements)
        return cls(unique_elements)

    @classmethod
    def from_json_file(cls, path: str):
        # Note: only works for matbench datasets
        file = open(path, 'r')
        dataset = json.load(file)
        file.close()
        try:
            data = dataset['data']
            return cls.from_matbench_dataset(data)
        except KeyError:
            print("The dataset does not contain the key 'data'. Check if the file is a matbench dataset.")
            raise 

    
    def save_atom_vectors(self, path: str, num_atom_features: Union[int, AnyStr] = 'all', test_uniqueness: bool = False):
        """
        Save the atom reprentation vectors in a json file for the whole dataset.
        """
        atom_vectors = {}
        dim_elements = self.total_elements # Number of unique elements in the data sets
        dim_properties = self.tableau_categories[-1] if num_atom_features == 'all' else num_atom_features
        ## we classify the elements in the following way:
        ## We enconde atomic (# dim_elements) distinct elements into vectors of dimension (# dim_properties) which encondes the physical properties
        enconding = one_hot(torch_arange(dim_elements), num_classes=dim_properties).numpy()
        print(f'    --> Dimension elements: {dim_elements}')
        print(f'    --> Dimension properties: {dim_properties}')
        print(f'    --> Enconding shape: {enconding.shape}')

        atom_vectors = {element: enconding[i,:].tolist() for i, element in enumerate(self.unique_elements)}

        if test_uniqueness:
            if len(atom_vectors) != self.total_elements:
                print("The number of elements in the dataset is not the same as the number of unique elements.")
                raise ValueError
            # Check that the vectors are unique
            seen = set()
            for key,vec in atom_vectors.items():
                # Convert list to tuple so it can be added to a set
                tuple_vec = tuple(vec)
                assert tuple_vec not in seen, f"Repeated list found in atom_vectors with key: {key}"
                seen.add(tuple_vec)

        with open(path, 'w') as file:
            json.dump(atom_vectors, file)

        