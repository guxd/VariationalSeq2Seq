import sys
import torch 
import torch.utils.data as data
import tables
import numpy as np
import unicodedata
import re
import string
import json
import collections

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def textproc(s):
    s=s.lower()
    s=s.replace('\'s ','is ')
    s=s.replace('\'re ','are ')
    s=s.replace('\'m ', 'am ')
    s=s.replace('\'ve ', 'have ')
    s=s.replace('\'ll ','will ')
    s=s.replace('n\'t ', 'not ')
    s=s.replace(' wo not',' will not')
    s=s.replace(' ca not',' can not')
    s=re.sub('[\!;-]+','',s)
    s=re.sub('\.+','.',s)
    if s.endswith(' .'):
        s=s[:-2]
    s=re.sub('\s+',' ',s)
    s=s.strip()
    return s

# convert opensubtitle data into TFRecord, and save the dict (int -> word) to json file
   

def create_dict(train_file, vocab_size):
    file=open(train_file,'r')
    counter = collections.Counter()
    for i, qaline in enumerate(file):
        line = qaline.translate(string.maketrans("", ""), string.punctuation)
        if line == "":
            break
        line=textproc(line)
        words = line.split()
        counter.update(words)
        if i % process_batch_size == 0 and i:
            print(str(i))
    file.close()

    dict = {'UNK': 2, '<SOS>':1, '<EOS>': 0}
    count=counter.most_common(vocab_size - 3)  # minus 1 for UNK
    for word, _ in count:
        if word=='':
            continue
        dict[word] = len(dict)
    return dict
    

def binarize(load_path, save_path, vocab):
    print("binarizing..")
    load_file = open(load_path, "r")
    qa_idxs=[]
    for i, qa_sent in enumerate(load_file):
        if i % process_batch_size == 0 and i:
            print(str(i))
        line = qa_sent.translate(string.maketrans("", ""), string.punctuation)
        if line == "":
            break
        line = line.strip("\r\n").split("\t")
        question = textproc(line[0]).split() + ["<EOS>"]
        answer =  textproc(line[1]).split() + ["<EOS>"]
        q_idx = [vocab.get(word,vocab['UNK']) for word in question]
        a_idx=[vocab.get(word,vocab['UNK']) for word in answer]
        qa_idxs.append([q_idx,a_idx])
        
    load_file.close()
    save_hdf5(qa_idxs, save_path)
    
    
class Index(tables.IsDescription):
    pos = tables.UInt32Col()
    q_len = tables.UInt32Col()
    a_len = tables.UInt32Col()
def save_hdf5(qa_idxs, filename):
    '''save the processed data into a hdf5 file'''
    print("writing hdf5..")
    f = tables.open_file(filename, 'w')
    filters = tables.Filters(complib='blosc', complevel=5)
    earrays = f.create_earray(f.root, 'sentences', tables.Int16Atom(),shape=(0,),filters=filters)
    indices = f.create_table("/", 'indices', Index, "a table of indices and lengths")
    count = 0
    pos = 0
    for qa in qa_idxs:
        q=qa[0]
        a=qa[1]
        earrays.append(np.array(q))
        earrays.append(np.array(a))
        ind = indices.row
        ind['pos'] = pos
        ind['q_len'] = len(q)
        ind['a_len'] = len(a)
        ind.append()
        pos += len(q)+len(a)
        count += 1
        if count % 1000000 == 0:
            print count
            sys.stdout.flush()
            indices.flush()
        elif count % 100000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    f.close()
    
def load_hdf5(self,idxfile):
    """read training sentences(list of int array) from a hdf5 file"""  
    table = tables.open_file(idxfile)
    data, index = (table.get_node('/sentences'),table.get_node('/indices'))
    data_len = index.shape[0]
    offset = 0
    print("{} entries".format(data_len))
    questions = []
    answers=[]
    while offset < data_len:
        pos, q_len, a_len =  index[offset]['pos'], index[offset]['q_len'], index[offset]['a_len']
        offset += 1
        questions.append(data[pos:pos + q_len].astype('int64'))
        answers.append(data[pos+q_len:pos+q_len+a_len].astype('int64'))
    table.close()
    return questions, answers


if __name__ == "__main__":
    data_path="./data/"
    train_file_in = data_path+"train.txt"
    valid_file_in = data_path+"valid.txt"
    train_file_out = data_path+"train.h5"
    valid_file_out = data_path+"valid.h5"
    # int -> word dict
    dict_path = data_path+"vocab.json"
    # how many words should be added into dict
    vocab_size = 30000
    # larger batch size speeds up the process but needs larger memory
    process_batch_size = 3000000
    
    print ("creating dictionary...")
    vocab=create_dict(train_file_in, vocab_size)
    dict_file = open(dict_path, "w")
    dict_file.write(json.dumps(vocab))

    print("processing training data...")
    binarize(train_file_in, train_file_out, vocab)

    print("processing valid data...")
    binarize(valid_file_in, valid_file_out, vocab)
    
    
#     print("reading...")
#     TRAIN_FILE=data_path+'valid.h5'
#     batch_size=20
#     dataset = UbuntuDataset(TRAIN_FILE)
#     data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                            batch_size=batch_size, 
#                                            shuffle=True,
#                                            num_workers=2)
#     data_iter=iter(data_loader)
#     batch=data_iter.next()





