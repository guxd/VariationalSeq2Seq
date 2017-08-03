import sys
import torch 
import torch.utils.data as data
import tables
import json
import numpy as np


class UbuntuDataset(data.Dataset):
    def __init__(self, filepath, max_seq_len):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.max_seq_len=max_seq_len
        
        print("loading data...")
        table = tables.open_file(filepath)
        self.data = table.get_node('/sentences')
        self.index = table.get_node('/indices')
        self.data_len = self.index.shape[0]
        print("{} entries".format(self.data_len))



    def __getitem__(self, offset):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        pos, q_len, a_len =  self.index[offset]['pos'], self.index[offset]['q_len'], self.index[offset]['a_len']
        question=self.data[pos:pos + q_len].astype('int64')
        answer=self.data[pos+q_len:pos+q_len+a_len].astype('int64')
        
        ## Padding ##
        if len(question)<self.max_seq_len:
            question=np.append(question, [0]*self.max_seq_len)    
        question=question[:self.max_seq_len]
        question[-1]=0
        if len(answer)<self.max_seq_len:
            answer=np.append(answer,[0]*self.max_seq_len)
        answer=answer[:self.max_seq_len]
        question[-1]=0
        
        ## get real seq len
        q_len=min(int(q_len),self.max_seq_len) # real length of question for training
        a_len=min(int(a_len),self.max_seq_len) 
        return question, answer, q_len, a_len

    def __len__(self):
        return self.data_len


def load_dict(filename):
    return json.loads(open(filename, "r").readline())



if __name__ == '__main__':
    
    input_dir='../../resources/data/opensub/'
    VALID_FILE=input_dir+'train.h5'
    valid_set=UbuntuDataset(VALID_FILE)
    valid_data_loader=torch.utils.data.DataLoader(dataset=valid_set,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)
    vocab = load_dict(input_dir+'vocab.json')
    ivocab = {v: k for k, v in vocab.items()}
    #print ivocab
    k=0
    for qapair in valid_data_loader:
        k+=1
        if k>20:
            break
        decoded_words=[]
        idx=qapair[0].numpy().tolist()[0]
        print idx
        for i in idx:
            decoded_words.append(ivocab[i])
        question = ' '.join(decoded_words)
        decoded_words=[]
        idx=qapair[1].numpy().tolist()[0]
        for i in idx:
            decoded_words.append(ivocab[i])
        answer=' '.join(decoded_words)
        print('<', question)
        print('>', answer)
