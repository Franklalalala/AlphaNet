import torch
import time
from torch import Tensor
from typing import Optional
from alphanet.models.alphanet import AlphaNet
from alphanet.models.graph import process_positions_and_edges
from alphanet.config import AlphaConfig
class AlphaNetWrapper(torch.nn.Module):
    def __init__(
        self,
        config: AlphaConfig
    ):
        super(AlphaNetWrapper, self).__init__()
        self.model = AlphaNet(config)
        self.cutoff = config.cutoff
        self.compute_forces = config.compute_forces
        self.compute_stress = config.compute_stress
        self.use_pbc = config.use_pbc
        self.precision =  torch.float32 if config.dtype == "32" else torch.float64
    def forward(self, 
            pos: Tensor,
    z: Tensor,
    batch: Tensor,
    natoms: Tensor,
    cell: Optional[Tensor] = None,
    prefix: str ='infer'):
        
        processed_data = process_positions_and_edges(pos,z,natoms,batch,cell, compute_forces=self.compute_forces, compute_stress=self.compute_stress, use_pbc=self.use_pbc, cutoff=self.cutoff, dtype = self.precision)
        output = self.model(processed_data, prefix)
        return output
    