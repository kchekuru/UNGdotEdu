import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from ase.io import read
from ase.neighborlist import neighbor_list
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class AtomicChargeDataset(InMemoryDataset):
    def __init__(self, root, poscar_dir, charge_dir, cutoff=7.0):
        self.poscar_dir = poscar_dir
        self.charge_dir = charge_dir
        self.cutoff = cutoff
        self.element_encoder = OneHotEncoder(sparse_output=False)
        self.element_encoder.fit(np.array(['Ba','O']).reshape(-1,1))
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['atomic_charge.pt']

    def process(self):
        data_list = []
        for poscar_name in sorted(os.listdir(self.poscar_dir)):
            if not poscar_name.startswith('CONFIG_'): continue
            atoms = read(os.path.join(self.poscar_dir, poscar_name), format='vasp')
            x = torch.tensor(self.element_encoder.transform(
                np.array(atoms.get_chemical_symbols()).reshape(-1,1)), dtype=torch.float)

            i, j, d = neighbor_list('ijD', atoms, self.cutoff, self_interaction=True)
            edge_index = torch.tensor(np.vstack((i, j)), dtype=torch.long)
            edge_attr = torch.tensor(d, dtype=torch.float)
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(1)

            charge_name = poscar_name.replace('CONFIG_', 'CHARGE_')
            y = torch.tensor(np.loadtxt(
                os.path.join(self.charge_dir, charge_name)), dtype=torch.float).view(-1, 1)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, name=poscar_name)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])