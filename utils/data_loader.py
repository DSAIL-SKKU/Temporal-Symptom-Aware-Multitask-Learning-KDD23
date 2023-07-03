import torch
from torch.utils.data import Dataset
import torch.nn as nn
from datetime import datetime
import numpy as np


def pad_collate_reddit(batch):    
    s_y = [item[0] for item in batch]
    b_y = [item[1] for item in batch]
    tweets = [torch.nan_to_num(item[2]) for item in batch] # 각자 잘하는 embedding이 달라서! 
    timestamp = [item[3] for item in batch]
    user_id = [item[4] for item in batch]
    
    #torch.nan_to_num(a)
    post_num = [len(x) for x in b_y]

    b_y = nn.utils.rnn.pad_sequence(b_y, batch_first=True, padding_value=0)
    tweets = nn.utils.rnn.pad_sequence(tweets, batch_first=True, padding_value=0)
    timestamp = nn.utils.rnn.pad_sequence(timestamp, batch_first=True, padding_value=0)
    
    
    
    post_num = torch.tensor(post_num)
    s_y = torch.tensor(s_y)
    user_id = torch.tensor(user_id)

    return [s_y, b_y, post_num, tweets,timestamp,user_id]

def get_timestamp(x):
    def change_utc(x):
        try:
            x = str(datetime.fromtimestamp(int(x)/1000))
            return x
        except:
            return str(x)

    timestamp = [datetime.timestamp(datetime.strptime(change_utc(t),"%Y-%m-%d %H:%M:%S")) for t in x]
    time_interval = (timestamp[-1] - np.array(timestamp))
    return time_interval


class RedditDataset(Dataset):
    def __init__(self, s_y, b_y, tweets,timestamp,user_id, days=30):
        super().__init__()
        self.s_y = s_y
        self.b_y = b_y
        self.tweets = tweets
        self.timestamp = timestamp
        self.user_id = user_id
        
        self.days = days

    def __len__(self):
        return len(self.s_y)

    def __getitem__(self, item):
        s_y = torch.tensor(self.s_y[item], dtype=torch.long)
        user_id = torch.tensor(self.user_id[item], dtype=torch.long)
        
        if self.days > len(self.tweets[item]):
            b_y = torch.tensor(self.b_y[item], dtype=torch.long) 
            tweets = torch.tensor(self.tweets[item], dtype=torch.float32)
            timestamp = get_timestamp(self.timestamp[item])
            timestamp = torch.tensor(timestamp, dtype=torch.float32)

        else:
            b_y = torch.tensor(self.b_y[item][:self.days], dtype=torch.long)
            tweets = torch.tensor(self.tweets[item][:self.days], dtype=torch.float32)
            timestamp = get_timestamp(self.timestamp[item][:self.days])
            timestamp = torch.tensor(timestamp, dtype=torch.float32)
        
        return [s_y, b_y, tweets,timestamp,user_id]


