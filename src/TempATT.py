import os
import pandas as pd
import numpy as np
import argparse

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule
from transformers import AdamW

from sklearn.model_selection import StratifiedGroupKFold
#from imblearn.over_sampling import RandomOverSampler

from utils.loss import loss_function
from utils.data_loader import RedditDataset, pad_collate_reddit
from utils.evaluation import *
from src.attention import Attention

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

class TempATT(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        # config:
        self.args = args
        self.config = config

        #model
        self.embed_type = self.config['embed_type'] + "_" + str(self.config['hidden_dim'])
        self.embed_layer = nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'])
        self.lstm = nn.LSTM(input_size=self.config['hidden_dim'],
                             hidden_size=int(self.config['hidden_dim']/2),
                             num_layers=2,
                             bidirectional=True)
        
        self.time_var = nn.Parameter(torch.randn((2)), requires_grad=True)
        self.atten = Attention(self.config['gpu'],self.config['hidden_dim'], batch_first=True)  # 2 is bidrectional
        self.dropout = nn.Dropout(self.config['dropout'])

        # suicide
        self.fc_1 = nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'])
        self.fc_2 = nn.Linear(self.config['hidden_dim'], self.config['s_y_num'])
        
        # aux
        self.b_decoder = nn.Linear(self.config['hidden_dim'], self.config['b_y_num'])
        
        # unweighted loss
        self.log_vars = nn.Parameter(torch.randn((self.args.task_num)))
        
    def forward(self, s_y, b_y, p_num, tweets,timestamp):        
        #lstm
        x = self.dropout(tweets)   
        
        # aux
        b_out = self.b_decoder(x) 
        logits_b = nn.utils.rnn.pack_padded_sequence(b_out, p_num.cpu(), batch_first=True, enforce_sorted=False)[0] 
        b_y = nn.utils.rnn.pack_padded_sequence(b_y, p_num.cpu(), batch_first=True, enforce_sorted=False)[0] 
        b_loss = nn.MultiLabelSoftMarginLoss(weight=None,reduction='mean')(logits_b, b_y)
        
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
        
        logits_s = self.fc_2(self.fc_1(self.dropout(h)))
        #logits_s = logits_s.view(-1, self.s_y_num)
        s_loss = loss_function(logits_s, s_y, self.config['loss'], self.config['s_y_num'], 1.8)

        # multi task loss            
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
        self.s_y_col = "fu_" + str(self.config['af']) + "_su_y"
        if self.config['s_y_num'] == 3:
            df[self.s_y_col] = df[self.s_y_col].apply(lambda x: 2 if x in [2,3] else x)
        elif self.config['s_y_num'] == 2:
            df[self.s_y_col] = df[self.s_y_col].apply(lambda x: 1 if x in [1,2,3] else x)
         
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.config['random_seed']) 
        for i,(train_idxs, test_idxs) in enumerate(cv.split(df, df[self.s_y_col], df['author'])):
            if i == self.config['n_fold']:
                break
        self.df_train = df.iloc[train_idxs]
        self.df_test = df.iloc[test_idxs]            
        print(f'# of train:{len(self.df_train)}, val:0, test:{len(self.df_test)}')
        
        # ros = RandomOverSampler(random_state=2023)
        # df_train, y_res = ros.fit_resample(df_train, df_train[self.s_y_col].tolist())
        
    def train_dataloader(self):
        self.train_data = RedditDataset(
            self.df_train[self.s_y_col].values, 
            self.df_train['cur_bp_y'].values, 
            self.df_train[self.embed_type].values,
            self.df_train["created_utc"].values,
            self.df_train['user_id'].values
        )
        return DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            collate_fn=pad_collate_reddit,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):
        self.test_data = RedditDataset(
            self.df_test[self.s_y_col].values, 
            self.df_test['cur_bp_y'].values, 
            self.df_test[self.embed_type].values,
            self.df_test["created_utc"].values,
            self.df_test['user_id'].values
        )
        return DataLoader(
            self.test_data,
            batch_size=self.args.batch_size,
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

        return {
            'loss': loss,
            's_true': s_true,
            's_preds': s_preds,
            'b_true': b_true, 
            'b_preds':b_preds,
            'user_id':user_id
        }

    def test_epoch_end(self, outputs):

        evaluation(self.config,outputs, 'fs','s_true', 's_preds','user_id')
        evaluation(self.config,outputs, 'bd','b_true', 'b_preds','user_id')        


