
import math
from math import pi
from typing import Optional, Tuple, List, NamedTuple
from typing import Literal
import torch
from torch import nn
from torch import Tensor
from torch.nn import Embedding
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from alphanet.models.graph import GraphData

class rbf_emb(nn.Module):
    '''
    modified: delete cutoff with r
    '''

    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        self.pi = pi
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value)) ** -2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        rbounds = 0.5 * \
                  (torch.cos(dist * self.pi / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).float()
        return rbounds * torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class NeighborEmb(MessagePassing):
    propagate_type = {
        'x': Tensor,
        'norm': Tensor
    }
    
    def __init__(self, hid_dim: int):
        super(NeighborEmb, self).__init__(aggr='add')
        self.embedding = nn.Embedding(95, hid_dim)
        self.hid_dim = hid_dim
        self.ln_emb = nn.LayerNorm(hid_dim, elementwise_affine=False)

    def forward(
        self,
        z: Tensor,
        s: Tensor,
        edge_index: Tensor,
        embs: Tensor
    ) -> Tensor:
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)
        s = s + s_neighbors
        return s

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, self.hid_dim) * x_j


class S_vector(MessagePassing):
    propagate_type = {
        'x': Tensor,
        'norm': Tensor
    }
    
    def __init__(self, hid_dim: int):
        super(S_vector, self).__init__(aggr='add')
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=False),
            nn.SiLU())

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        edge_index: Tensor,
        emb: Tensor
    ) -> Tensor:
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v
        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)

class EquiMessagePassing(MessagePassing):

    propagate_type = {
        'xh': Tensor,
        'vec': Tensor,
        'rbfh_ij': Tensor,
        'r_ij': Tensor
    }

    def __init__(
            self,
            hidden_channels,
            num_radial,
            hidden_channels_chi=96,
            head: int = 16,
            chi1: int = 32,
            chi2: int = 8,
            has_dropout_flag: bool = False,
            has_norm_before_flag=True,
            has_norm_after_flag=False,
            reduce_mode='sum',
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    ):
        super(EquiMessagePassing, self).__init__(aggr="add", node_dim=0)

        self.device = device
        self.reduce_mode = reduce_mode
        self.chi1 = chi1
        self.chi2 = chi2
        self.head = head
        self.hidden_channels = hidden_channels
        self.hidden_channels_chi = hidden_channels_chi
        self.scale = nn.Linear(self.hidden_channels, self.hidden_channels_chi * 2)
        self.num_radial = num_radial
        self.dir_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_channels + self.num_radial, self.hidden_channels * 3), nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3), )

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3)
        self.x_layernorm = nn.LayerNorm(hidden_channels)
        self.diagonal = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels_chi // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels_chi // 2, self.chi2),
        )
        self.has_dropout_flag = has_dropout_flag
        self.has_norm_before_flag = has_norm_before_flag
        self.has_norm_after_flag = has_norm_after_flag

        if self.has_norm_after_flag:
            self.dx_layer_norm = nn.LayerNorm(self.chi1)
        if self.has_norm_before_flag:
            self.dx_layer_norm = nn.LayerNorm(self.chi1 + self.hidden_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.diachi1 = torch.nn.Parameter(torch.randn((self.chi1), device=self.device))
        self.scale2 = nn.Sequential(
            nn.Linear(self.chi1, hidden_channels//2),
        )

        self.kernel_real = torch.nn.Parameter(torch.randn((self.head + 1, (self.hidden_channels_chi) // self.head, self.chi2), device=self.device))
        self.kernel_imag = torch.nn.Parameter(torch.randn((self.head + 1, (self.hidden_channels_chi) // self.head, self.chi2), device=self.device))
        
        self.fc_mps = nn.Linear(self.chi1, self.chi1)#.to(torch.cfloat)
        self.fc_dx = nn.Linear(self.chi1, hidden_channels)#.to(torch.cfloat)
        self.dia = nn.Linear(self.chi1, self.chi1)#.to(torch.cfloat)
      
        self.unitary = torch.nn.Parameter(torch.randn((self.chi1, self.chi1), device=self.device))
        self.activation = nn.SiLU()

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()
        

        nn.init.xavier_uniform_(self.dir_proj[0].weight)
        self.dir_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dir_proj[2].weight)
        self.dir_proj[2].bias.data.fill_(0)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        edge_rbf: Tensor,
        weight: Tensor,
        edge_vector: Tensor,
        rope: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if rope is not None:
            real, imag = torch.split(x, [self.hidden_channels//2, self.hidden_channels//2], dim=-1)
            dy_pre = torch.complex(real=real, imag=imag)
            dy_pre = dy_pre* rope
            x = torch.cat([dy_pre.real, dy_pre.imag], dim=-1)
        xh = self.x_proj(self.x_layernorm(x))

        rbfh = self.rbf_proj(edge_rbf)
        weight = self.dir_proj(weight)
        rbfh = rbfh * weight
        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
            # rotation = unitary,
        )
        if self.has_norm_before_flag:
            dx = self.dx_layer_norm(dx)

        dx, dy = torch.split(dx, [self.chi1, self.hidden_channels], dim=-1)

        if self.has_norm_after_flag:
            dx = self.dx_layer_norm(dx)

        dx = self.scale2(dx)

        dx = torch.complex(torch.cos(dx), torch.sin(dx))
        
        return dx, dy, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        
        real, imagine = torch.split(self.scale(x), self.hidden_channels_chi, dim=-1)
        real = real.reshape(x.shape[0], self.head, (self.hidden_channels_chi) // self.head)
        imagine = imagine.reshape(x.shape[0], self.head, (self.hidden_channels_chi) // self.head)
        if self.has_dropout_flag:
            real = self.dropout(real)
            imagine = self.dropout(imagine)

        # complex invariant quantum state
        phi = torch.complex(real, imagine)
        q = phi
        a = torch.ones(q.shape[0], 1, (self.hidden_channels_chi) // self.head, device=self.device, dtype=torch.complex32)
        kernel = (torch.complex(self.kernel_real, self.kernel_imag) / math.sqrt((self.hidden_channels) // self.head)).expand(q.shape[0], -1, -1, -1)
        equation = 'ijl, ijlk->ik'
        conv = torch.einsum(equation, torch.cat([a, q], dim=1), kernel.to(torch.complex32))
        a = 1.0 * self.activation(self.diagonal(rbfh_ij))
        b = a.unsqueeze(-1) * self.diachi1.unsqueeze(0).unsqueeze(0) + torch.ones(kernel.shape[0], self.chi2, self.chi1, device=self.device)
        dia = self.dia(b)
        equation = 'ik,ikl->il'
        kernel = torch.einsum(equation, conv, dia.to(torch.complex32))
        kernel_real,kernel_imag = kernel.real,kernel.imag
        kernel_real,kernel_imag  = self.fc_mps(kernel_real),self.fc_mps(kernel_imag)
        kernel = torch.angle(torch.complex(kernel_real, kernel_imag))
        agg = torch.cat([kernel, x], dim=-1)
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return agg, vec

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.reduce_mode)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class FTE(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec = self.vec_proj(vec)
        vec1, vec2 = torch.split(
            vec, self.hidden_channels, dim=-1
        )

        scalar = torch.norm(vec1, dim=-2, p=1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = vec_dot * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, scalar], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2

        return dx, dvec


class aggregate_pos(MessagePassing):

    def __init__(self, aggr='mean'):
        super(aggregate_pos, self).__init__(aggr=aggr)

    def forward(self, vector, edge_index):
        v = self.propagate(edge_index, x=vector)

        return v


class AlphaNet(nn.Module):
    
    def __init__(self, config, device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")):
        super(AlphaNet, self).__init__()

        self.device = device
        self.eps = config.eps
        self.num_layers = config.num_layers
        self.hidden_channels = config.hidden_channels
        self.a = nn.Parameter(torch.ones(108) * config.a)
        self.b = nn.Parameter(torch.ones(108) * config.b)
        self.cutoff = config.cutoff
        self.readout = config.readout
        self.chi1 = config.main_chi1
        self.use_sigmoid = config.use_sigmoid
        self.num_targets = config.output_dim if config.output_dim != 0 else 1
        self.compute_forces = config.compute_forces
        self.compute_stress = config.compute_stress
        self.z_emb_ln = nn.LayerNorm(config.hidden_channels, elementwise_affine=False)
        self.z_emb = Embedding(95, config.hidden_channels)
        self.kernel1 = torch.nn.Parameter(torch.randn((config.hidden_channels, self.chi1 * 2), device=self.device))
        self.radial_emb = rbf_emb(config.num_radial, config.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(config.num_radial, config.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels, config.hidden_channels))
        self.pi = pi 
        self.neighbor_emb = NeighborEmb(config.hidden_channels)
        self.S_vector = S_vector(config.hidden_channels)
        self.lin = nn.Sequential(
            nn.Linear(3, config.hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels // 4, 1))
        
        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()
        self.kernels_real = []
        self.kernels_imag = []
        
        for _ in range(config.num_layers):
            self.message_layers.append(
                EquiMessagePassing(
                    hidden_channels=config.hidden_channels,
                    num_radial=config.num_radial,
                    head=config.head,
                    chi2=config.chi2,
                    chi1=config.mp_chi1,
                    has_dropout_flag=config.has_dropout_flag,
                    has_norm_before_flag=config.has_norm_before_flag,
                    has_norm_after_flag=config.has_norm_after_flag,
                    hidden_channels_chi=config.hidden_channels_chi,
                    device=device,
                    reduce_mode=config.reduce_mode
                )
            )
            self.FTEs.append(FTE(config.hidden_channels))
            
            kernel_real = torch.randn((config.hidden_channels, self.chi1, self.chi1), device=self.device)
            kernel_imag = torch.randn((config.hidden_channels, self.chi1, self.chi1), device=self.device)
            self.kernels_real.append(kernel_real)
            self.kernels_imag.append(kernel_imag)
            
        self.kernels_real = torch.nn.Parameter(torch.stack(self.kernels_real))
        self.kernels_imag = torch.nn.Parameter(torch.stack(self.kernels_imag))
        
        self.last_layer = nn.Linear(config.hidden_channels, self.num_targets)
        self.last_layer_quantum = nn.Linear(self.chi1 * 2, self.num_targets)
        
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.reset_parameters()

    def reset_parameters(self):
        self.z_emb.reset_parameters()
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        self.last_layer.reset_parameters()
        
        for layer in self.radial_lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data: GraphData, prefix: str):
        pos = data.pos
        batch = data.batch
        z = data.z.long()
        edge_index = data.edge_index
        dist = data.edge_attr
        vecs = data.edge_vec

        z_emb = self.z_emb_ln(self.z_emb(z))
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        rbounds = 0.5 * (torch.cos(dist * self.pi / self.cutoff) + 1.0)
        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden

        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)
        
        j = edge_index[0]
        i = edge_index[1]
        edge_diff = vecs
        edge_diff = edge_diff / (dist.unsqueeze(1) + self.eps)
        mean = scatter(pos[edge_index[0]], edge_index[1], reduce='mean', dim=0)
        
        edge_cross = torch.cross(pos[i]-mean[i], pos[j]-mean[i])
        edge_vertical = torch.cross(edge_diff, edge_cross)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)

        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (self.lin(torch.permute(scalrization1, (0, 2, 1))) + 
                  torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1) / math.sqrt(self.hidden_channels)
        scalar4 = (self.lin(torch.permute(scalrization2, (0, 2, 1))) + 
                  torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1) / math.sqrt(self.hidden_channels)
        
        edge_weight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edge_weight = torch.cat((edge_weight, radial_hidden, radial_emb), dim=-1)
        
        equation = 'ik,bi->bk'
        quantum = torch.einsum(equation, self.kernel1, z_emb)
        real, imagine = torch.split(quantum, self.chi1, dim=-1)
        quantum = torch.complex(real, imagine)

        rope = None
        for id, (message_layer, fte) in enumerate(zip(self.message_layers, self.FTEs)):
            if rope is None:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, None)
            else:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, rope)
            
            s = s + ds
            vec = vec + dvec
            
            kernel_real = self.kernels_real[id]
            kernel_imag = self.kernels_imag[id]
            equation = 'ikl,bi,bl->bk'
            kerneli = torch.complex(kernel_real, kernel_imag)
            quantum = torch.einsum(equation, kerneli, s.to(torch.cfloat), quantum)
            quantum = quantum / quantum.abs().to(torch.cfloat)
            
            ds, dvec = fte(s, vec)
            s = s + ds
            vec = vec + dvec

        s = self.last_layer(s) + self.last_layer_quantum(torch.cat([quantum.real, quantum.imag], dim=-1)) / self.chi1
        
        if s.dim() == 2:
            s = (self.a[z].unsqueeze(1) * s + self.b[z].unsqueeze(1))
        elif s.dim() == 1:
            s = (self.a[z] * s + self.b[z]).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected shape of s: {s.shape}")
        
        s = scatter(s, batch, dim=0, reduce=self.readout)
        
        if self.use_sigmoid:
            s = torch.sigmoid((s - 0.5) * 5)
        if self.use_sigmoid:
            s = torch.sigmoid((s - 0.5) * 5)
            
        if self.compute_forces and self.compute_stress:
            
            if data.displacement is not None:
              stress, forces = self.cal_stress_and_force(s, pos, data.displacement, data.cell, prefix)
              stress = stress.view(-1, 3)
            else:
                stress = None
                forces = None
            return s, forces, stress
        elif self.compute_forces:
            forces = self.cal_forces(s, data.pos, prefix)
            return s, forces, None
        return s, None, None
    
    def cal_forces(self, energy, positions, prefix: str = 'infer'):
 
        graph = (prefix == "train")
        grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(energy)])
        forces = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=graph,
            allow_unused=True
        )[0]
        assert forces is not None, "Gradient should not be None"
        return -forces

    def cal_stress_and_force(self, energy: Tensor,positions: Tensor, displacement: Optional[Tensor], cell: Tensor, prefix: str) -> Tuple[Tensor, Tensor]:
        if displacement is None:
         raise ValueError("displacement cannot be None for stress calculation")      
        graph = (prefix == "train")
        grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(energy)])
        output = torch.autograd.grad(
            [energy],
            [displacement, positions],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=graph,
            allow_unused=True
        )
        virial = output[0]
        force =output[1]
        assert virial is not None, "Virial tensor should not be None"
        assert force is not None, "Forces tensor should not be None"
        return -virial, -force



