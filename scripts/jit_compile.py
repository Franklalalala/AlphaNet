import torch
import os
from alphanet.models.model import AlphaNetWrapper
from alphanet.config import All_Config

def freeze_and_save_model(config_path, checkpoint_path, save_path):
    """
    Freeze and save a PyTorch model.
    
    Args:
        config_path (str): Path to the JSON configuration file.
        checkpoint_path (str): Path to the model checkpoint file.
        save_path (str): Path to save the frozen model.
    """
    # Load configuration
    config = All_Config().from_json(config_path)
    
    # Initialize model
    model = AlphaNetWrapper(config.model)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(ckpt, strict=False)
    
    # Freeze and optimize model
    frozen_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    
    # Save the frozen model
    frozen_model.save(save_path)
    print(f"Model saved to {save_path}")

# Example usage
if __name__ == "__main__":
    config_path = "water.json"
    checkpoint_path = "water.ckpt"
    save_path = "water.pt"
    
    freeze_and_save_model(config_path, checkpoint_path, save_path)
