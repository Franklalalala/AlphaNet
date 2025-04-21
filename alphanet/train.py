import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from alphanet.data import get_pic_datasets
from alphanet.models.model import AlphaNetWrapper
from alphanet.mul_trainer import Trainer

def run_training(config1,config2):
    
    train_dataset, valid_dataset, test_dataset = get_pic_datasets(root='dataset/', name=config1.dataset_name,config = config1)
    force_std = torch.std(train_dataset.data.force).item()
  
    energy_peratom = torch.sum(train_dataset.data.y).item()/torch.sum(train_dataset.data.natoms).item()
    config1.a = force_std
    config1.b = energy_peratom
    print(config1.a, config1.b)    
    model = AlphaNetWrapper(config1)
    print(model.model.a, model.model.b)
    if config1.dtype == "64":
      model = model.double()
    #strategy = DDPStrategy(num_nodes=config["hardware"]["num_nodes"]) if config["hardware"]["num_nodes"] > 1 else "auto"
    checkpoint_callback = ModelCheckpoint(
        dirpath=config1.train.save_dir,
        filename='{epoch}-{val_loss:.4f}-{val_energy_loss:.4f}-{val_force_loss:.4f}',
        save_top_k=-1,
    every_n_epochs=1,  
    save_on_train_epoch_end=True,  
        monitor='val_loss',
        mode='min'
    )
 
    trainer = pl.Trainer(
        devices=config2["num_devices"],
        num_nodes=config2["num_nodes"],
        strategy='ddp_find_unused_parameters_true',
        accelerator="gpu" if config2["num_devices"] > 0 else "cpu",
        max_epochs=config1.epochs,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        gradient_clip_val=0.5,
        default_root_dir=config1.train.save_dir,
        accumulate_grad_batches=config1.accumulation_steps,
    )
    
    model = Trainer(config1, model, train_dataset, valid_dataset, test_dataset)
    trainer.fit(model, ckpt_path=config2["ckpt_path"] if config2["resume"] else None)

