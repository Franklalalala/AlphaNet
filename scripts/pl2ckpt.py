import torch

def update_and_save_state_dict(input_checkpoint_path, output_checkpoint_path, prefix_to_replace, replacement):
    """
    Update the keys in a PyTorch model's state dictionary and save the updated dictionary.

    Args:
        input_checkpoint_path (str): Path to the input checkpoint file.
        output_checkpoint_path (str): Path to save the updated checkpoint file.
        prefix_to_replace (str): Prefix in the keys to be replaced.
        replacement (str): String to replace the prefix with.
    """
    # Load the checkpoint
    model = torch.load(input_checkpoint_path)
    state_dict = model['state_dict']
    print(f"Original state_dict keys: {list(state_dict.keys())[:5]}...")  # Print first 5 keys for reference

    # Update state_dict keys
    new_state_dict = {k.replace(prefix_to_replace, replacement): v for k, v in state_dict.items()}
    
    # Save the updated state_dict
    torch.save(new_state_dict, output_checkpoint_path)
    print(f"State_dict keys updated and saved to {output_checkpoint_path} successfully.")

# Example usage
if __name__ == "__main__":
    input_checkpoint_path = 'gap/epoch=9-val_loss=9.7872-val_energy_mae=0.0000-val_force_mae=0.0000.ckpt'
    output_checkpoint_path = 'water.ckpt'
    prefix_to_replace = 'l.model'
    replacement = 'l'

    update_and_save_state_dict(input_checkpoint_path, output_checkpoint_path, prefix_to_replace, replacement)
