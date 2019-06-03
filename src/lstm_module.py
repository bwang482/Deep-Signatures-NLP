import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)
        
    def forward(self, x):
        x = x.to(device)
        output, hidden = self.rnn(x)
        return self.fc(output[:,-1,:])

class RecurrentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bidirectional=True, dropout=0.5, lstm_flag=True):
        super(RecurrentModel, self).__init__()

        self.lstm_flag = lstm_flag

        # Building LSTM: batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        if lstm_flag:
            self.mod = nn.LSTM(input_dim, 
                               hidden_dim, 
                               layer_dim, 
                               bidirectional=bidirectional, 
                               dropout=dropout, 
                               batch_first=True).to(device)

            self.fc = nn.Linear(hidden_dim * 2, output_dim).to(device)
            self.dropout = nn.Dropout(dropout).to(device)
        else:
            self.mod = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True).to(device)
            self.fc = nn.Linear(hidden_dim, output_dim).to(device)      

    def forward(self, x):
        
        x = x.to(device)

        if self.lstm_flag:
            out, (hn, cn) = self.mod(x)
            hn = self.dropout(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1))
            return self.fc(hn.squeeze(0))
        
        out, hn = self.mod(x)
        return self.fc(out[:, -1, :])