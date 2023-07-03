import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from collections import Counter
import pickle
import random
import argparse
import time
from datetime import datetime
import math

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from transformers import AdamW

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold
#from imblearn.over_sampling import RandomOverSampler

from utils.loss import loss_function, true_metric_loss
from utils.utils import class_FScore, gr_metrics, make_31, splits
from utils.data_loader import RedditDataset, pad_collate_reddit
from src.attention import Attention

class Arg:
    random_seed: int = 2022  # Random Seed
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



class Model(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        
        # config:
        self.args = args
        self.config = config
        self.seed = self.config['random_seed']
        self.save = self.config['save']
        self.bf = self.config['bf']
        self.af = self.config['af']      
        self.batch_size = self.args.batch_size
        self.embed_type = self.config['embed_type']
        self.hidden_size = self.config['hidden_dim'] # BERT-base: 768, BERT-large: 1024, BERT paper setting
        self.embed_type = f'{self.embed_type}_{str(self.hidden_size)}'
        
        ## 0. target label
        # bp classification (auxilary task)
        self.b_y_col = 'cur_bp_y' 
        self.b_y_names = ['bp_no','bp_remission', 'bp_manic', 'bp_irritability', 
                          'bp_anxiety', 'bp_depressed', 'bp_psychosis', 'bp_somatic']
        self.b_y_num = 8

        
        # future si risk prediction (main task)
        self.s_y_col = f'fu_{self.af}_su_y'
        self.s_y_num = self.config['s_y_num']
        
        if self.s_y_num == 4:
            self.s_y_names = ['su_indicator', 'su_ideation','su_behavior', 'su_attempt']
        elif self.s_y_num == 3:
            self.s_y_names = ['su_indicator', 'su_ideation','su_behav + att']
        elif self.s_y_num == 2:
            self.s_y_names = ['su_indicator', 'su_id+beh+att']
        
        self.embed_layer = nn.Linear(self.hidden_size, self.hidden_size)

        self.lstm = nn.LSTM(input_size=self.hidden_size,
                             hidden_size=int(self.hidden_size/2),
                             num_layers=2,
                             bidirectional=True)
        
        self.time_var = nn.Parameter(torch.randn((2)), requires_grad=True)
        
        self.atten = Attention(self.config['gpu'],self.hidden_size, batch_first=True)  # 2 is bidrectional
        self.dropout = nn.Dropout(self.config['dropout'])

        # suicide
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.s_y_num)
        
        # aux
        self.b_decoder = nn.Linear(self.hidden_size, self.b_y_num)
        
        # unweighted loss
        self.task_num = self.args.task_num
        self.log_vars = nn.Parameter(torch.randn((self.task_num)))
        
    def forward(self, s_y, b_y, p_num, tweets,timestamp):        
        
        #lstm
        x = self.dropout(tweets)   
        
        # aux
        b_out = self.b_decoder(x) 
        logits_b = nn.utils.rnn.pack_padded_sequence(b_out, p_num.cpu(), batch_first=True, enforce_sorted=False)[0] 

        # main
        x = nn.utils.rnn.pack_padded_sequence(x, p_num.cpu(), batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        #time sensitive attention suicide
        timestamp = torch.exp(self.time_var[0]) *timestamp + self.time_var[0]
        timestamp = torch.sigmoid(timestamp+ self.time_var[1]) #.size()
        x = x+ x*timestamp.unsqueeze(-1)
        h, att_score = self.atten(x, p_num.cpu())  # skip connect
        
        
        #reddit model        
        if h.dim() == 1:
            h = h.unsqueeze(0) 
        
        logits = self.fc_2(self.fc_1(self.dropout(h)))
        logits_s = logits.view(-1, self.s_y_num)

        s_loss = loss_function(logits_s, s_y, self.config['loss'], self.s_y_num, 1.8)

        # multi task loss            
        b_y = nn.utils.rnn.pack_padded_sequence(b_y, p_num.cpu(), batch_first=True, enforce_sorted=False)[0] 
        b_loss = nn.MultiLabelSoftMarginLoss(weight=None,reduction='mean')(logits_b, b_y)
        
        s_prec = torch.exp(-self.log_vars[0])
        s_loss = s_prec*s_loss + self.log_vars[0]
        
        b_prec = torch.exp(-self.log_vars[1])
        b_loss = b_prec*b_loss + self.log_vars[1]
        
        total_loss =  s_loss  + b_loss
        
        return total_loss, b_loss, logits_s, timestamp,att_score, b_y,logits_b
    

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.001)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess_dataframe(self):   
        
        data_path = './dataset/sample_data.pkl'       
        df = pd.read_pickle(data_path)
        
        # class split
        if self.config['s_y_num'] == 3:
            df[self.s_y_col] = df[self.s_y_col].apply(lambda x: 2 if x in [2,3] else x)
        elif self.config['s_y_num'] == 2:
            df[self.s_y_col] = df[self.s_y_col].apply(lambda x: 1 if x in [1,2,3] else x)
        
         
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed) 
        for i,(train_idxs, test_idxs) in enumerate(cv.split(df, df[self.s_y_col], df['author'])):
            if i == self.config['n_fold']:
                break
        print("TRAIN:", len(train_idxs), " TEST: ", len(test_idxs))

        df_train = df.iloc[train_idxs]
        df_test = df.iloc[test_idxs]
        
        # ros = RandomOverSampler(random_state=2023)
        # df_train, y_res = ros.fit_resample(df_train, df_train[self.s_y_col].tolist())
 
        print(f'# of train:{len(df_train)}, val:0, test:{len(df_test)}')

        self.train_data = RedditDataset(df_train[self.s_y_col].values, 
                                        df_train[self.b_y_col].values, 
                                        df_train[self.embed_type].values,
                                       df_train["created_utc"].values,
                                       df_train['user_id'].values)
                                       
        
        self.test_data = RedditDataset(df_test[self.s_y_col].values, 
                                       df_test[self.b_y_col].values, 
                                       df_test[self.embed_type].values,
                                      df_test["created_utc"].values,
                                      df_test['user_id'].values)
        
    def train_dataloader(self):
        
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=pad_collate_reddit,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=pad_collate_reddit,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        s_y, b_y, p_num, tweets,timestamp,user_id = batch  
        loss, b_loss, logit,timestamp,att_score, b_true, b_pred= self(s_y, b_y, p_num, tweets,timestamp)    
        self.log("train_loss", loss)
            
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        s_y, b_y, p_num, tweets,timestamp,user_id = batch  
        loss, b_loss, logit,timestamp,att_score, b_true, b_pred= self(s_y, b_y, p_num, tweets,timestamp)         
                
        # preds
        s_true = list(s_y.cpu().numpy())
        s_preds = list(logit.argmax(dim=-1).cpu().numpy())
        
        b_true = list(b_true.cpu().numpy())
        b_pred = F.softmax(b_pred, dim=1)
        b_preds = np.array(b_pred.cpu()>0.14).astype(int)
        b_preds = list(b_preds)
        user_id = list(user_id.cpu().numpy())

#         pd.DataFrame(att_score.cpu().numpy(),index=user_id).to_csv(
#              f'./result/df_analysis/{datetime.now().__format__("%m%d_%H%M")}_class{self.s_y_num}_{self.bf}_{self.af}_{self.save}_att_{batch_idx}.csv') 


        return {
            'loss': loss,
            's_true': s_true,
            's_preds': s_preds,
            'b_true': b_true, 
            'b_preds':b_preds,
            'user_id':user_id
        }
                

    def test_epoch_end(self, outputs):
       
        # pred save
        def pred_save(_type,y_true_col,y_pred_col,label_names,user_id_col):
            y_true = []
            y_pred = []
            user_id = []

            for i in outputs:
                y_true += i[y_true_col]
                y_pred += i[y_pred_col]
                user_id += i[user_id_col]
            
            y_true = np.asanyarray(y_true)
            y_pred = np.asanyarray(y_pred)
            user_id = np.asanyarray(user_id)
        
            pred_dict = {}
        
            pred_dict['user_id']= user_id
            pred_dict['y_true']= y_true
            pred_dict['y_pred']= y_pred
            

            print("-------test_report-------")
            metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                                 target_names = label_names, 
                                                 output_dict=True)
            df_result = pd.DataFrame(metrics_dict).transpose() 
            pprint(df_result)
        
            print("-------save test_report-------")
            self.save_time = datetime.now().__format__("%m%d_%H%M%S%Z")
            self.save_path = f"./result/"
            Path(f"{self.save_path}/pred").mkdir(parents=True, exist_ok=True)
            
            df_result.to_csv(f'./result/{self.save_time}_{_type}_{self.bf}_{self.af}_{self.save}.csv')              
            
            with open(f'{self.save_path}pred/{self.save_time}_{_type}_{self.bf}_{self.af}_{self.save}_pred.pkl', "wb") as outfile:
                pickle.dump(pred_dict, outfile)   


        pred_save('fs','s_true', 's_preds',self.s_y_names,'user_id')
        pred_save('bd','b_true', 'b_preds',self.b_y_names,'user_id')        

    
def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything(config['random_seed'])
        
    model = Model(args,config) 
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
        deterministic=True, # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        # For GPU Setup
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
    parser.add_argument("--n_fold", type=int, default=4) 
    
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__) 

