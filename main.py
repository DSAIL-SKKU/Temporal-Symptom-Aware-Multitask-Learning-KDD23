import os
import argparse
import numpy as np
import random

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

## 추가
import configparser
import warnings
warnings.filterwarnings('ignore')

from src.TempATT import TempATT

def th_seed_everything(seed: int = 2023):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

class Arg:
    epochs: int = 1  # Max Epochs, BERT paper setting [3,4,5]
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    batch_size: int = 64 
    max_post_num = 30
    task_num: int = 2
            
def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything(config['random_seed'])
    th_seed_everything(config['random_seed'])
       
    # 일단 mood
    model = TempATT(args,config) 
    model.preprocess_dataframe()

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    print(":: Start Training ::")
    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback],
        enable_checkpointing = False,
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        deterministic=True,
        gpus=[config['gpu']] if torch.cuda.is_available() else None, 
        precision=16 if args.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model,dataloaders=model.test_dataloader())
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dropout", type=float, default=0.01,help="dropout probablity")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--gpu", type=int, default=1,  help="save fname")
    parser.add_argument("--random_seed", type=int, default=2022) 
    parser.add_argument("--bf", type=int, default=6) 
    parser.add_argument("--af", type=int, default=30) 
    parser.add_argument("--embed_type", type=str, default="sb") 
    parser.add_argument("--hidden_dim", type=int, default=1024) 
    parser.add_argument("--loss", type=str, default="oe") 
    parser.add_argument("--save", type=str, default="test") 
    parser.add_argument("--s_y_num", type=int, default=2) 
    parser.add_argument("--b_y_num", type=int, default=8) 
    parser.add_argument("--n_fold", type=int, default=4) 
    
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__) 

