# import standard libraries
import re
import random
import time
import math
import numpy as np
import numpy.random as npr

# torch
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# plotting libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from siglayer import backend

# global variables
MAX_LENGTH = 10
SOS_token = 0 # start-of-string token
EOS_token = 1 # end-of-string token

# Helper functions
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

## Sig Encoder
#class SigEncoder(nn.Module):
#    def __init__(self, input_size, embedding_size, projecting_size, sig_depth=2):
#        super(SigEncoder, self).__init__()

#        sig_dim = backend.sig_dim(projection_dim, sig_depth)

#        self.embedding = nn.Embedding(input_size, embedding_size)

#        self.projection = nn.Linear(embedding_size, projecting_size)

#        self.sig = backend.SigLayer(sig_depth)

#    def forward(self, input, hidden):
#        embedded = self.embedding(input).view(1, 1, -1) 
#        output = embedded
#        output, hidden = self.gru(output, hidden)
#        return output, hidden

# RNN Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        # note the recurrent step: the class calls itself via Inheritance
        super(EncoderRNN, self).__init__()
        
        # size of hidden state
        self.hidden_size = hidden_size
        
        # simple lookup table that stores embeddings of a fixed dictionary and size
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # multi-layer gated recurrent unit (GRU) RNN
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1) #like numpy reshape
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# RNN Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Attention Decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Training
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, 
          with_attention=True, teacher_forcing_ratio=0.5):
    
    # initialize the hidden state of the encoder
    encoder_hidden = encoder.initHidden()
    
    # initialize gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # initialize inputs sizes
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # initialize output size 
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # initial loss
    loss = 0

    # encoder inputs/outputs
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # initialize decoder input and hidden state
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    # train with Attention Decoder
    if with_attention:

        # random teacher forcing
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # with teacher forcing
        if use_teacher_forcing: 
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # feed the target as the next input

        # Without teacher forcing
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)

                # detach from history as input  and use its own predictions as the next input
                decoder_input = topi.squeeze().detach()  

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token: # break if we are at the end
                    break
    
    # train with standard RNN Decoder
    else:
        for di in range(target_length):
            # decoder takes as input its previous output and hidden state
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)

            # detach from history as input  and use its own predictions as the next input
            decoder_input = topi.squeeze().detach()  

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token: # break if we are at the end
                break

    # loss gradient
    loss.backward()
    
    # backpropagation through encoder and decoder
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH, with_attention=True):
    # we don't need backprop anymore in testing
    with torch.no_grad():
        # same working flow as in training
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []

        # evaluate with attention model
        if with_attention:
            decoder_attentions = torch.zeros(max_length, max_length)
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                # store decoder attention to see what matters
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
            # encoder hidden is the representation of a sentence!!!! We want to see that!
            return decoded_words, decoder_attentions[:di + 1], encoder_hidden

        # evaluate without attention model
        else:
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
            return decoded_words, encoder_hidden

# helper function for random model evaluation
def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=10, with_attention=True):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        if with_attention:
            output_words, attentions,_ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, with_attention=True)
        else:
            output_words,_ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, with_attention=False)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

# helper function for plotting results
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

# Call training function many times
def trainIters(encoder, decoder, n_iters, pairs, input_lang, output_lang, print_every=1000, plot_every=100, learning_rate=0.01, with_attention=True):
    
    # 1. Start timer
    start = time.time()
    plot_losses = []
    print_loss_total = 0  
    plot_loss_total = 0  

    # 2. Initialize optimizers and criterion
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    # 3. Create set of training pairs
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(n_iters)]
    
    # 4. The negative log likelihood loss
    criterion = nn.NLLLoss()

    # 5. Call train many times
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, with_attention=with_attention)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)