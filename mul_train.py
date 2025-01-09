import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from alphanet.data import get_pic_datasets
from alphanet.models.model import AlphaNetWrapper
from alphanet.config import All_Config
from alphanet.mul_trainer import Trainer
import os
def main():
    config = All_Config().from_json("OC2M-train.json")
    train_dataset, valid_dataset, test_dataset = get_pic_datasets(root='dataset/', name=config.data.dataset_name,
                                                                     train_size=config.data.train_size, valid_size=config.data.valid_size, 
                                                                     seed=config.data.seed, train_dataset=config.data.train_dataset, 
                                                                     valid_dataset=config.data.valid_dataset, test_dataset=config.data.test_dataset)
    force_std = torch.std(train_dataset.data.force).item()
    ENERGY_MEAN_TOTAL = 0
    FORCE_MEAN_TOTAL = 0
    NUM_ATOM = None

    for data in valid_dataset:
        energy = data.y
        force = data.force
        NUM_ATOM = force.size()[0]
        energy_mean = energy / NUM_ATOM
        ENERGY_MEAN_TOTAL += energy_mean

    ENERGY_MEAN_TOTAL /= len(train_dataset)
    
    config.model.a = force_std
    config.model.b = ENERGY_MEAN_TOTAL
    print(config.model.a)    
    model = AlphaNetWrapper(config.model)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.train.save_dir,
        filename='{epoch}-{val_loss:.4f}-{val_energy_mae:.4f}-{val_force_mae:.4f}',
        save_top_k=-1,
    every_n_epochs=1,  
    save_on_train_epoch_end=True,  
        monitor='val_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience= 50,
        mode='min'
    )

    trainer = pl.Trainer(
        devices=3,
        num_nodes=1,
        limit_train_batches=40000,
        accelerator='auto',
       #inference_mode=False,

    strategy='ddp_find_unused_parameters_true',    
    max_epochs=config.train.epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        default_root_dir=config.train.save_dir,
        logger=pl.loggers.TensorBoardLogger(config.train.log_dir),
        gradient_clip_val=0.5,
        accumulate_grad_batches=config.train.accumulation_steps
    )

    model = Trainer(config, model, train_dataset, valid_dataset, test_dataset)
    trainer.fit(model)#, ckpt_path = ckpt)
    trainer.test()

if __name__ == '__main__':
    main()
