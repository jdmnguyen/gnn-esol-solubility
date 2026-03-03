# src/dataset.py
import os
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from .featurization import smiles_to_mol, mol_to_graph_data_obj

class ESOLDataset(InMemoryDataset):
    def __init__(self, root="data", transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # Explicitly set weights_only=False for PyTorch 2.6+
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # We expect this file at data/raw/esol.csv
        return ["esol.csv"]

    @property
    def processed_file_names(self):
        return ["esol_pyg.pt"]

    @property
    def raw_dir(self):
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed")

    def download(self):
        # We already manually put esol.csv into data/raw/, so nothing to do.
        pass

    def process(self):
        # Read raw CSV
        raw_path = osp.join(self.raw_dir, "esol.csv")
        df = pd.read_csv(raw_path)

        data_list = []
        for i, row in df.iterrows():
            smiles = row["smiles"]
            y_value = float(row["logS"])
            mol = smiles_to_mol(smiles)
            data = mol_to_graph_data_obj(mol, y_value)
            data_list.append(data)

        # Save in PyG format
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    dataset = ESOLDataset(root="data")
    print(dataset)
    print("Example:", dataset[0])
