"""Experimental sequence baselines, not reproductions of named published models."""
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.memory = nn.LSTM(features,32,batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(32),nn.Dropout(.1),nn.Linear(32,1))

    def forward(self,x):
        x,_ = self.memory(x)
        return self.head(x[:,-1]).squeeze(-1)


class CNNClassifier(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(features,32,3,padding=1),nn.ReLU(),
                                 nn.Conv1d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(32,1)

    def forward(self,x):
        return self.head(self.net(x.transpose(1,2)).squeeze(-1)).squeeze(-1)


class PatchClassifier(nn.Module):
    def __init__(self,features,lookback):
        super().__init__()
        self.patch = min(4,lookback)
        self.length = (lookback//self.patch)*self.patch
        self.projection = nn.Linear(features*self.patch,32)
        self.position = nn.Parameter(torch.zeros(1,lookback//self.patch,32))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,dropout=.1,
                                                                         batch_first=True),num_layers=1)
        self.head = nn.Linear(32,1)

    def forward(self,x):
        # Attention only sees the completed historical window at the decision time.
        x = x[:,-self.length:]
        x = x.reshape(x.shape[0],self.length//self.patch,-1)
        x = self.encoder(self.projection(x)+self.position)
        return self.head(x.mean(dim=1)).squeeze(-1)
