

from __future__ import unicode_literals, print_function, division
from io import open

import string
import re
import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
######################################################################
# The Seq2Seq Model
# =================
class Encoder(nn.Module):
    def __init__(self, input_size , hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input):
        embedded = input#self.embedding(input)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        output, hidden = self.gru(embedded) # out: [b x seq x hid_sz*2] (biRNN)   hidden: [2 x batch_sz x hid_sz] (2=fw&bw)
        return output, hidden
           
class LatentVariation(nn.Module):
    def __init__(self, rnn_hid_size, z_hid_size):
        super(LatentVariation, self).__init__()
        self.z_hid_size=z_hid_size   
        
        self.context_to_mean=nn.Linear(rnn_hid_size, z_hid_size) # activation???
        ''' ??? no activation??? '''
        self.context_to_logvar=nn.Linear(rnn_hid_size, z_hid_size) 

    def forward(self, q, a): # q: batch_sz x 2*hid_sz
        #print(qa_enc)
        [batch_size,_]=q.size()
        context=torch.cat([q,a],1) #batch_sz x 4*hid_sz
        mean=self.context_to_mean(context)
        logvar = self.context_to_logvar(context) #batch_sz x z_hid_size
        std = torch.exp(0.5 * logvar)

        z = Variable(torch.randn([batch_size, self.z_hid_size]))
        z = z.cuda() if torch.cuda.is_available() else z
        z = z * std + mean   # [batch_sz x z_hid_sz]
        
        return z, mean, logvar
    
class Descriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Descriminator, self).__init__()
        
        self.W1=nn.Linear(hidden_size, hidden_size)
        self.W2=nn.Linear(hidden_size, 1)
        self.activation=nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, q, a): 
        qa=torch.cat([q, a],1)
        hid=self.activation(self.W1(qa))
        out=self.activation(self.W2(hid))
        out=self.sigmoid(out)
        return out
        

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size=vocab_size

        #self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()
        self.dropout=nn.Dropout(p=0.3)

    def forward(self, input, prev_state): #input: [batch_sz x 1 x emb_sz] (1=seq_len)  prev_state: [1 x batch_sz X hid_sz]
        batch_size, seq_len, emb_sz=input.size()
        output = input#self.dropout(input)#self.embedding(input) # output: [batch_sz x 1 x emb_sz] (1=seq_len)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, prev_state = self.gru(output, prev_state) # output: [batch_sz x 1 x hid_sz] (1=seq_len)
        #print(output.size())
        output = self.softmax(self.out(output.view(-1, self.hidden_size)))   # output: [batch_sz x hid_sz]
        output = output.view(batch_size, seq_len, self.vocab_size) # output: [batch_sz x 1 x vocab_sz] (1=seq_len)
        #print(output.size())
        hidden=prev_state
        return output, hidden

   # def initHidden(self):
    #    result = Variable(torch.zeros(1, 1, self.hidden_size))
    #    if torch.cuda.is_available():
    #        return result.cuda()
    #    else:
    #        return result


