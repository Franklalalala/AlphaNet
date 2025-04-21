import argparse
from alphanet.data import get_pic_datasets
from alphanet.config import All_Config
from alphanet.evaler import Evaluator
from torch_geometric.data import DataLoader
import torch

def main():
    parser = argparse.ArgumentParser(description='Evaluate a machine learning force field model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run evaluation on.')
    parser.add_argument('--plots_dir', type=str, default='./plots', help='Directory to save plots.')
    parser.add_argument('--data_root', type=str, default='dataset/', help='Root directory of the dataset.')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable tqdm progress bar.')
    args = parser.parse_args()

    config = All_Config().from_json(args.config)
    train_dataset, valid_dataset, test_dataset = get_pic_datasets(root='dataset/', name=config.dataset_name,config = config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers
    )

    evaluator = Evaluator(args.ckpt, config, device=args.device)
    evaluator.plot_energy_parity(
        train_loader,
        val_loader,
        test_loader,
        plots_dir=args.plots_dir,
        disable=args.disable_tqdm
    )
    evaluator.plot_force_parity(
        train_loader,
        val_loader,
        test_loader,
        plots_dir=args.plots_dir,
        disable=args.disable_tqdm
    )

if __name__ == '__main__':
    main()

