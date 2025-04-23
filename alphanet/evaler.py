import torch
import os
import matplotlib.pyplot as plt
from torch.autograd import grad
from tqdm import tqdm
from alphanet.models.model import AlphaNetWrapper
class Evaluator:
    def __init__(self, model_path, config, device='cpu'):
        self.device = torch.device(device)
        self.config = config
        self.model = None
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        

    def load_model(self, model_path):
        self.model = AlphaNetWrapper(self.config.model)
        checkpoint = torch.load(model_path)
        
        self.model.load_state_dict(checkpoint)


    def plot_energy_parity(self, train_loader, val_loader, test_loader, plots_dir=None, disable=False):
        datasets = {'Train': train_loader, 'Validation': val_loader, 'Test': test_loader}
        colors = {'Train': '#1f77b4', 'Validation': '#ff7f0e', 'Test': '#2ca02c'}

        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.5)

        for name, loader in datasets.items():
            preds_energy = torch.Tensor([])#.to(self.device)
            targets_energy = torch.Tensor([])#.to(self.device)

            for batch_data in tqdm(loader, disable=disable):
                batch_data = batch_data.to(self.device)
                batch_data.pos.requires_grad = True 
                if self.config.model.use_pbc:
                   model_outputs =  self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, batch_data.cell, "infer")
                else:
                  model_outputs = self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, prefix ="infer")
                
                energy, _, _ = model_outputs
                
                energy = energy.squeeze()
                preds_energy = torch.cat([preds_energy.cpu(), (energy / batch_data.natoms).detach().cpu()], dim=0)
                targets_energy = torch.cat([targets_energy.cpu(), batch_data.y.cpu() / batch_data.natoms.cpu()], dim=0)

            deviation = torch.abs(preds_energy - targets_energy)
            threshold = 100 * torch.sqrt(torch.mean((preds_energy - targets_energy) ** 2)).item()
            mask_deviation = deviation < threshold
            mask_nan = ~torch.isnan(preds_energy) & ~torch.isnan(targets_energy)
            mask = mask_deviation & mask_nan
            preds_energy_filtered = preds_energy[mask]
            targets_energy_filtered = targets_energy[mask]
            energy_mae_filtered = torch.mean(torch.abs(preds_energy_filtered - targets_energy_filtered)).item()
            energy_rmse_filtered = torch.sqrt(torch.mean((preds_energy_filtered - targets_energy_filtered) ** 2)).item()
            plt.scatter(
                targets_energy_filtered.cpu().numpy(),
                preds_energy_filtered.cpu().numpy(),
                alpha=0.7,
                label=f'{name}: MAE={energy_mae_filtered:.4f}, RMSE={energy_rmse_filtered:.4f}',
                color=colors[name],
                s=20
            )

        min_energy = targets_energy.min().cpu().numpy()
        max_energy = targets_energy.max().cpu().numpy()
        plt.plot([min_energy, max_energy], [min_energy, max_energy], 'k--', lw=2, label='Ideal')
        plt.xlabel('True Energy per Atom', fontsize=17)
        plt.ylabel('Predicted Energy per Atom', fontsize=17)
        plt.title('Energy Parity Plot', fontsize=19)
        plt.legend(fontsize=17, loc='upper left', bbox_to_anchor=(0.1, 0.95), ncol=1, framealpha=0.8)
        plt.tight_layout()

        if plots_dir is None:
            plots_dir = './plots'
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'energy_parity_plot_combined.png'), dpi=1000)
        plt.close()

    def plot_force_parity(self, train_loader, val_loader, test_loader, plots_dir=None, disable=False):
        datasets = {'Train': train_loader, 'Validation': val_loader, 'Test': test_loader}
        colors = {'Train': '#1f77b4', 'Validation': '#ff7f0e', 'Test': '#2ca02c'}

        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.5)

        for name, loader in datasets.items():
            preds_force = torch.Tensor([])#.to(self.device)
            targets_force = torch.Tensor([])#.to(self.device)

            for batch_data in tqdm(loader, disable=disable):
                batch_data = batch_data.to(self.device)
                batch_data.pos.requires_grad = True 
                if self.config.model.use_pbc:
                   model_outputs =  self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, batch_data.cell, "infer")
                else:
                   model_outputs = self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, prefix ="infer")
                if self.config.compute_forces:
                    _, force,_ = model_outputs

                    if torch.sum(torch.isnan(force)) != 0:
                        mask = ~torch.isnan(force)
                        force = force[mask].reshape((-1, 3))
                        batch_data.force = batch_data.force[mask].reshape((-1, 3))

                    preds_force = torch.cat([preds_force.cpu(), force.detach().cpu()], dim=0)
                    targets_force = torch.cat([targets_force.cpu(), batch_data.force.cpu()], dim=0)

            deviation = torch.abs(preds_force - targets_force)
            threshold = 100 * torch.sqrt(torch.mean((preds_force - targets_force) ** 2)).item()
            mask = deviation < threshold
            preds_force_filtered = preds_force[mask]
            targets_force_filtered = targets_force[mask]
            force_mae_filtered = 0.5*torch.mean(torch.abs(preds_force_filtered - targets_force_filtered)).item()
            force_rmse_filtered = 0.5*torch.sqrt(torch.mean((preds_force_filtered - targets_force_filtered) ** 2)).item()

            plt.scatter(
                targets_force_filtered.cpu().numpy(),
                preds_force_filtered.cpu().numpy(),
                alpha=0.7,
                label=f'{name}: MAE={force_mae_filtered:.4f}, RMSE={force_rmse_filtered:.4f}',
                color=colors[name],
                s=20
            )

        min_force = targets_force.min().cpu().numpy()
        max_force = targets_force.max().cpu().numpy()
        plt.plot([min_force, max_force], [min_force, max_force], 'k--', lw=2, label='Ideal')

        plt.xlabel('True Force', fontsize=17)
        plt.ylabel('Predicted Force', fontsize=17)
        plt.title('Force Parity Plot', fontsize=19)
        plt.legend(fontsize=17, loc='upper left', bbox_to_anchor=(0.1, 0.95), ncol=1, framealpha=0.8)
        plt.tight_layout()

        if plots_dir is None:
            plots_dir = './plots'
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'force_parity_plot_combined.png'), dpi=500)
        plt.close()

    def evaluate(self, data_path, plots_dir=None, disable=False):
        train_loader, val_loader, test_loader = self.load_data(data_path)
        self.plot_energy_parity(train_loader, val_loader, test_loader, plots_dir, disable)
        self.plot_force_parity(train_loader, val_loader, test_loader, plots_dir, disable)

if __name__ == '__main__':
    model_path = 'path_to_your_model.ckpt'  
    data_path = 'path_to_your_data'       
    plots_dir = './plots'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    evaluator = Evaluator(model_path, device)
    evaluator.evaluate(data_path, plots_dir)

