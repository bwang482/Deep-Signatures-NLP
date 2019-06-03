import candle
import torch
import torch.nn as nn
import torch.nn.functional as F

from siglayer import backend, modules, examples

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepSigSentimentNet(nn.Module):    
    def __init__(self, input_dim, embedding_dim, output_dim, sig_depth=2, 
                 augment_layer_sizes=(16, 2), augment_kernel_size=8, lengths=(5, 10), 
                 strides=(2, 5), adjust_lengths=(0, 0), layer_sizes_s=((16,), (16, 16)), 
                 memory_sizes=(8, 8), augment_include_original=True, augment_include_time=True,
                 hidden_output_sizes=(8,), optim_embedding=False):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.windowed = examples.create_windowed((output_dim,), sig=True, sig_depth=sig_depth, 
                                                 final_nonlinearity=lambda x: x,
                                                 augment_layer_sizes=augment_layer_sizes, 
                                                 augment_kernel_size=augment_kernel_size,
                                                 lengths=lengths, strides=strides, adjust_lengths=adjust_lengths, 
                                                 layer_sizes_s=layer_sizes_s, memory_sizes=memory_sizes,
                                                 augment_include_original=augment_include_original, 
                                                 augment_include_time=augment_include_time,
                                                 hidden_output_sizes=hidden_output_sizes)
        
    def forward(self, text):
        #text = [sent len, batch size]
        text = text[0].permute(1, 0)
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)
        return F.log_softmax(self.windowed(embedded))


class SigSentimentNet(nn.Module):    
    def __init__(self, input_dim, embedding_dim, output_dim, layer_sizes=(16, 16),
                 augment_layer_sizes = (50, 50, 12), augment_kernel_size=1, 
                 augment_include_original=False, augment_include_time=False, sig_depth=2):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.sig_complex = examples.create_simple(output_dim, sig=True, sig_depth=sig_depth, 
                                                  final_nonlinearity=lambda x: x,
                                                  augment_layer_sizes=augment_layer_sizes, 
                                                  augment_kernel_size=augment_kernel_size, 
                                                  augment_include_original=augment_include_original, 
                                                  augment_include_time=augment_include_time, 
                                                  layer_sizes=layer_sizes)
        
    def forward(self, text):
        #text = [sent len, batch size]
        text = text[0].permute(1, 0)
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch_size, emb dim, sent len]
        return F.log_softmax(self.sig_complex(embedded))

class RNN(nn.Module):
    """In some frameworks one must feed the initial hidden state, $h_0$, 
       into the RNN, however in PyTorch, if no initial hidden state is passed 
       as an argument it defaults to a tensor of all zeros."""
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 optim_embedding=False):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        """output is the concatenation of the hidden state from every time step, 
           whereas hidden is simply the final hidden state"""

        #text = [sent len, batch size]
        embedded = self.embedding(text[0])
        #embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        return F.log_softmax(self.fc(hidden.squeeze(0)))


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, optim_embedding=False):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        # apply LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output = [sent len, batch size, hid dim * num directions] ..... output over padding tokens are zero tensors
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        #hidden = [batch size, hid dim * num directions]
            
        return F.log_softmax(self.fc(hidden.squeeze(0)))


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, optim_embedding=False):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.gru = nn.GRU(embedding_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        #text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        # apply GRU layer
        packed_output, hidden = self.gru(packed_embedded)
        #hidden = [num layers * num directions, batch size, hid dim]]

        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output = [sent len, batch size, hid dim * num directions] ...... output over padding tokens are zero tensors

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        #hidden = [batch size, hid dim * num directions]
            
        return F.log_softmax(self.fc(hidden.squeeze(0)))


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx, optim_embedding=False):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text[0].permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return F.log_softmax(self.fc(cat))