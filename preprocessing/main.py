import os
from preprocessing.atom_loader import AtomLoader

def main(working_dir: str, dataset: str):
    """
    Main function to create the atom features for a given dataset.
    working_dir: Path to the working directory.
    dataset: Name of the dataset.
    """
    atom_loader = AtomLoader.from_json_file(
        os.path.join(working_dir, dataset, 'raw', f'matbench_{dataset}.json')
    )
    atom_loader.save_atom_vectors(
        os.path.join(working_dir, dataset, 'processed', f'{dataset}_atom_features.json'),
        num_atom_features ='all',
        test_uniqueness=True
    )

if __name__ == '__main__':
    STORE = 'path/to/storage/' #change as needed
    working_dir = os.path.join(STORE, 'datasets/matbench')
    datasets = ['mp_e_form', 'mp_gap', 'mp_is_metal',  'perovskites'] # add more if needed
    for dataset in datasets:
        print('---------------------------------')
        print(f'Processing {dataset}')
        main(working_dir, dataset)
        print(f'Finished processing {dataset}')
        print('---------------------------------')
