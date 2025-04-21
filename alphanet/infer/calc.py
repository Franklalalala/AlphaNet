import torch
import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from alphanet.models.model import AlphaNetWrapper

import numpy as np
import torch
from ase.calculators.calculator import Calculator


class AlphaNetCalculator(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, ckpt_path, config, device='cpu', precision='32', **kwargs):
        Calculator.__init__(self, **kwargs)
        if ckpt_path.endswith('ckpt'):
          self.model = AlphaNetWrapper(config).to(torch.device(device))
          self.model.load_state_dict(torch.load(ckpt_path))        
        elif ckpt_path.endswith('pt'):
           self.model = torch.load(ckpt_path,map_location=torch.device(device))
        else:
          raise ValueError("Unknown checkpoint format") 
        
        self.device = torch.device(device)
        self.precision = torch.float32 if precision == "32" else torch.float64
        if precision == "64":
         self.model.double()
        self.model.to(self.device)
        self.config = config

    def calculate(self, atoms=None, properties=None, system_changes=[]):
        Calculator.calculate(self, atoms, properties, system_changes)
        properties = properties or ['energy']
        

        z = torch.tensor(
            [atomic_numbers[atom.symbol] for atom in atoms], 
            dtype=torch.long, 
            device=self.device
        )
        pos = torch.tensor(
            atoms.get_positions(), 
            dtype=self.precision, 
            device=self.device, 
            requires_grad=(self.config.compute_forces)  
        )
        cell = torch.tensor(
            atoms.get_cell(complete=True), 
            dtype=self.precision, 
            device=self.device
        ) if atoms.pbc.any() else None
        natoms = torch.tensor(
            [len(atoms)], 
            dtype=torch.int64, 
            device=self.device
        )
        batch = torch.zeros_like(z).to(self.device)
        
        with torch.set_grad_enabled(self.config.compute_forces):
            energy, forces, stress =  self.model(pos,z,batch,natoms, cell, "infer")
        
       
        self.results['energy'] = energy.detach().cpu().item()
        self.results['free_energy'] = self.results['energy']  
        
        if forces is not None:
            self.results['forces'] = forces.detach().cpu().numpy()
        
        if  stress is not None:
            stress_matrix = stress.detach().cpu().numpy()
           
            self.results['stress'] = np.array([
                stress_matrix[0, 0],  # xx
                stress_matrix[1, 1],  # yy
                stress_matrix[2, 2],  # zz
                0.5 * (stress_matrix[1, 2] + stress_matrix[2, 1]),  # yz
                0.5 * (stress_matrix[0, 2] + stress_matrix[2, 0]),  # xz
                0.5 * (stress_matrix[0, 1] + stress_matrix[1, 0])   # xy
            ])

