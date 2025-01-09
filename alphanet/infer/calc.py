import torch
import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers

class AlphaNetCalculator(Calculator):
    implemented_properties = ['energy','free_energy', 'forces', 'stress']

    def __init__(self, model, device='cpu', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model.to(device)
        self.device = device

    def calculate(self, atoms=None, properties=['energy', 'free_energy','forces', 'stress'], system_changes=[]):
        Calculator.calculate(self, atoms, properties, system_changes)
        z = torch.tensor([atomic_numbers[atom.symbol] for atom in atoms], dtype=torch.long).to(self.device)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32).to(self.device)
        pos.requires_grad = True
        batch = torch.zeros_like(z).to(self.device)
        cell = torch.tensor(atoms.get_cell(complete=True), dtype=torch.float32).to(self.device)
      
        natoms = torch.tensor([len(atoms)], dtype=torch.int32).to(self.device)
        output = self.model(pos,z,batch,natoms,cell, "infer")
        
        energy, forces, stress = output
        
        self.results['energy'] = energy.detach().cpu().numpy()[0][0]
        if 'forces' in properties:
            self.results['forces'] = forces.detach().cpu().numpy()

        if 'stress' in properties:
            stress_matrix = stress.detach().cpu().numpy()
            stress_ase = np.array([stress_matrix[0, 0],  # xx
                                   stress_matrix[1, 1],  # yy
                                   stress_matrix[2, 2],  # zz
                                   stress_matrix[1, 2],  # yz
                                   stress_matrix[0, 2],  # xz
                                   stress_matrix[0, 1]])  # xy
            self.results['stress'] = stress_ase
        if 'free_energy' in properties:
            self.results['free_energy']=energy.detach().cpu().item()

