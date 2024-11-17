import torch

model = torch.load('t2_15/epoch=0-val_loss=13.1895-val_energy_mae=0.0000-val_force_mae=0.0000.ckpt')
state_dict = model['state_dict']
print(state_dict.keys())
new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
torch.save(new_state_dict, 't2_15/t2.ckpt')

print("State_dict keys updated and saved successfully.")

