
from __future__ import unicode_literals, print_function, division
from io import open
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data import load_dict,UbuntuDataset
from helper import showPlot, timeSince, sent2indexes, indexes2sent, sortbatch
from model import Encoder,Decoder, LatentVariation

use_cuda = torch.cuda.is_available()

SOS_token = 1
EOS_token = 0
UNK_token = 2

MAX_SEQ_LEN = 20

######################################################################
# Training the Model

def _train_step(q_batch, a_batch, q_lens, a_lens, embedder, encoder, hidvar, decoder, 
                embedder_optimizer, encoder_optimizer, hidvar_optimizer, decoder_optimizer,
                criterion, kl_anneal_weight, p_teach_force = 0.5):
    '''
    Train one instance
    '''
    embedder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    hidvar_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    batch_size=len(q_batch)
    
    q = Variable(q_batch) # shape: [batch_sz x seq_len]
    q = q.cuda() if use_cuda else q
    a = Variable(a_batch)
    a = a.cuda() if use_cuda else a
    #max_a_len = max(a_lens)
    #q_lens=Variable(torch.LongTensor(q_lens)) # shape: [batch_sz x 1]
    #a_lens=Variable(torch.LongTensor(a_lens))
    q_emb=embedder(q)
    a_emb=embedder(a)
    _, q_enc = encoder(q_emb)
    _, a_enc = encoder(a_emb)
    q_enc=torch.cat([q_enc[0],q_enc[1]],1)  # batch_sz x 2*hid_sz
    """!!!!!! to be optimized !!!!!!!"""
    a_enc=torch.cat([a_enc[0],a_enc[1]],1)
    z, mean, logvar = hidvar(q_enc, a_enc) 
    kl_loss = (-0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
    
    
    decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size]).view(batch_size,1))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input # [batch_sz x 1] (1=seq_len)
    decoder_input=embedder(decoder_input)
    decoder_hidden = torch.cat([q_enc,z],1).unsqueeze(0) # [1 x batch_sz x hid_sz] (1=n_layers)
    use_teach_force = True if random.random() < p_teach_force else False
    out=None
    for di in range(a.size(1)):
        decoder_output, decoder_hidden = decoder( # decoder_output: [batch_sz x 1 x vocab_sz]
                    decoder_input, decoder_hidden)
        if di==0: # first step
            out=decoder_output
        else:
            out=torch.cat([out, decoder_output],1)
            
        if use_teach_force:# Teacher forcing: Feed the target as the next input
            decoder_input = a[:,di].unsqueeze(1)  # Teacher forcing
            #print("decoder_input_sz_1:")
            #print(decoder_input.size())
        else: # Without teacher forcing: use its own predictions as the next input
            topi = decoder_output[:,-1].max(1,keepdim=True)[1] # topi:[batch_sz x 1] indexes of predicted words
            decoder_input = topi#Variable(torch.LongTensor(ni))
            #ni = topi.cpu().numpy().squeeze().tolist() #!!
            #if ni == EOS_token:
            #    break
        decoder_input=embedder(decoder_input)
    #print(out)
    target = pack_padded_sequence(a,a_lens.tolist(), batch_first=True)[0]
    #print (target)
    packed_out = pack_padded_sequence(out,a_lens.tolist(), batch_first=True)[0]
    #print (packed_out)
    decoder_loss = criterion(packed_out, target)
    #print (decoder_loss)
    #decoder_loss=decoder_loss/a_lens
    
    loss=79*decoder_loss+ kl_loss*kl_anneal_weight
    
    loss.backward()
    
    embedder_optimizer.step()
    encoder_optimizer.step()
    hidvar_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], kl_loss.data[0], decoder_loss.data[0]


######################################################################
# The whole training process

def train(embedder, encoder, hidvar, decoder, data_loader, vocab, n_iters,  model_dir, p_teach_force=0.5,
          save_every=5000, sample_every=100, print_every=10, plot_every=100, learning_rate=0.00005):
    start = time.time()
    print_time_start=start
    plot_losses = []
    print_loss_total, print_loss_kl, print_loss_decoder = 0., 0., 0.  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    embedder_optimizer=optim.Adam(embedder.parameters(), lr=learning_rate)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    hidvar_optimizer = optim.Adam(hidvar.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(weight=None, size_average=True) #, ignore_index=EOS_token) #average over a batch, ignore EOS

    data_iter = iter(data_loader)
    
    for it in range(1, n_iters + 1):
        q_batch,a_batch, q_lens, a_lens = data_iter.next() 
        
        q_batch,a_batch, q_lens, a_lens = sortbatch(q_batch, a_batch, q_lens, a_lens)# !!! important for pack sequence
             # sort sequences according to their lengthes in descending order
        
        kl_anneal_weight = (math.tanh((it - 3500)/1000) + 1)/2
     
        total_loss, kl_loss, decoder_loss = _train_step(q_batch, a_batch, q_lens, a_lens,
                                                        embedder, encoder, hidvar, decoder, 
                                                        embedder_optimizer, encoder_optimizer, 
                                                        hidvar_optimizer, decoder_optimizer,
                                                        criterion, kl_anneal_weight, p_teach_force)
            
        print_loss_total += total_loss
        print_loss_kl+=kl_loss
        print_loss_decoder+=decoder_loss
        plot_loss_total += total_loss
        if it % save_every ==0:
            if not os.path.exists('%slatentvar_%s/' % (model_dir, str(it))):
                os.makedirs('%slatentvar_%s/' % (model_dir, str(it)))
            torch.save(f='%slatentvar_%s/embedder.pckl' % (model_dir, str(it)),obj=embedder)
            torch.save(f='%slatentvar_%s/encoder.pckl' % (model_dir,str(it)),obj=encoder)
            torch.save(f='%slatentvar_%s/hidvar.pckl' % (model_dir,str(it)),obj=hidvar)
            torch.save(f='%slatentvar_%s/decoder.pckl' % (model_dir,str(it)),obj=decoder)
        if it % sample_every == 0:
            samp_idx=np.random.choice(len(q_batch),4) #pick 4 samples
            for i in samp_idx:
                question, target = q_batch[i].view(1,-1), a_batch[i].view(1,-1)
                sampled_sentence = sample(embedder, encoder, hidvar, decoder, question, vocab)
                ivocab = {v: k for k, v in vocab.items()}
                print('question: %s'%(indexes2sent(question.squeeze().numpy(),ivocab, ignore_tok=EOS_token)))
                print('target: %s'%(indexes2sent(target.squeeze().numpy(), ivocab, ignore_tok=EOS_token)))
                print('predicted: %s'%(sampled_sentence))
        if it % print_every == 0:
            print_loss_total = print_loss_total / print_every
            print_loss_kl = print_loss_kl / print_every
            print_loss_decoder = print_loss_decoder / print_every
            print_time=time.time()-print_time_start
            print_time_start=time.time()
            print('iter %d/%d  step_time:%ds  total_time:%s tol_loss: %.4f kl_loss: %.4f dec_loss: %.4f'%(it, n_iters, 
                  print_time, timeSince(start, it/n_iters), print_loss_total, print_loss_kl, print_loss_decoder))
            print_loss_total, print_loss_kl, print_loss_decoder=0,0,0
        if it % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    #showPlot(plot_losses)


def sample(embedder, encoder, hidvar, decoder, question, vocab, max_length=MAX_SEQ_LEN):
    ivocab = {v: k for k, v in vocab.items()}
    q = Variable(question) # shape: [batch_sz (=1) x seq_len]
    q = q.cuda() if use_cuda else q
    
    q_emb=embedder(q)
    _, q_enc = encoder(q_emb)
    q_enc=torch.cat([q_enc[0],q_enc[1]],1)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]).view(1,1))  # SOS: [batch_sz(=1) x seq_len(=1)]
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_input = embedder(decoder_input)
    
    z = Variable(torch.randn([1, hidvar.z_hid_size]))
    z = z.cuda() if use_cuda else z
    #print(z.size())
    #print(q_enc.size())
    
    decoder_hidden = torch.cat([q_enc,z],1).unsqueeze(0) # [1 x batch_sz x hid_sz] 

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topi = decoder_output[:,-1].max(1,keepdim=True)[1] # topi:[batch_sz(=1) x 1] indexes of predicted words
        ni = topi.squeeze().data.cpu().numpy()[0] #!!
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(ivocab[ni])
        
        decoder_input = topi
        decoder_input = embedder(decoder_input)

    return ' '.join(decoded_words)




if __name__ == '__main__':
    
    input_dir='./data/'
    model_dir=input_dir+'models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    emb_size = 256 
    rnn_hid_size = 1024
    z_hid_size = 1024  # Get mean and variance of latent variable z
    n_words=30000
    batch_size=32
    n_iters=120000
    save_every=5000
    sample_every=100
    print_every=10
    learning_rate=0.0005
    
    p_teach_force=0.5 # prob of using teacher forcing, i.e., inputing decoder with ground truth or generated words 
    
    reload_from=-1
    
    embedder=nn.Embedding(n_words, emb_size, padding_idx=EOS_token)
    encoder = Encoder(emb_size, rnn_hid_size)
    hidvar= LatentVariation(2*2*rnn_hid_size, z_hid_size) # 2*2->[q_hid_fw;q_hid_bk;a_hid_fw;a_hid_bk]
    decoder = Decoder(emb_size, 2*rnn_hid_size + z_hid_size, n_words,1)
    if reload_from>0:# if using from previous data
        embedder= torch.load(f='%slatentvar_%s/embedder.pckl' %(model_dir, str(reload_from)))
        encoder = torch.load(f='%slatentvar_%s/encoder.pckl' % (model_dir, str(reload_from)))
        hidvar = torch.load(f='%slatentvar_%s/hidvar.pckl' % (model_dir, str(reload_from)))
        decoder = torch.load(f='%slatentvar_%s/decoder.pckl' % (model_dir, str(reload_from)))

    if use_cuda:
        embedder=embedder.cuda()
        encoder = encoder.cuda()
        hidvar = hidvar.cuda() #!!!!!
        decoder = decoder.cuda()
    
    TRAIN_FILE=input_dir+'train.h5'
    train_set = UbuntuDataset(TRAIN_FILE, max_seq_len=20)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=1 # multiple num_workers could introduce error (conflict?) 
                                          )
    vocab = load_dict(input_dir+'vocab.json')
    
    train(embedder, encoder, hidvar, decoder, train_data_loader, vocab, n_iters, model_dir, p_teach_force, 
          save_every=save_every, sample_every=sample_every, print_every=print_every, learning_rate=learning_rate)

    
    VALID_FILE=input_dir+'valid.h5'
    valid_set=UbuntuDataset(VALID_FILE, max_seq_len=20)
    valid_data_loader=torch.utils.data.DataLoader(dataset=valid_set,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)
    
    ## validation
    for qapair in valid_data_loader:
        #print('>', qapair[0])
        #print('=', qapair[1])
        output_sentence = sample(embedder, encoder, hidvar, decoder, qapair[0], vocab)
        print('<', output_sentence)
        #print('')
        
        
    # test
    question="how are you"
    question_indexes=sent2indexes(question, vocab)
    output_sentence = sample(embedder, encoder, hidvar, decoder, question_indexes, vocab)
    print(output_sentence)
    

