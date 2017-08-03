######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math
import numpy as np
import torch


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


def sent2indexes(sentence, vocab):
    return [vocab[word] for word in sentence.split(' ')]
def indexes2sent(indexes, ivocab, ignore_tok=-1):
    indexes=filter(lambda i: i!=ignore_tok, indexes)
    return ' '.join([ivocab[idx] for idx in indexes])

def sortbatch(q_batch, a_batch, q_lens, a_lens):
    """ 
    sort sequences according to their lengthes in descending order
    """
    maxlen_q = max(q_lens)
    maxlen_a = max(a_lens)
    q=q_batch[:,:maxlen_q-1]
    a=a_batch[:,:maxlen_a-1]
    sorted_idx = torch.LongTensor(a_lens.numpy().argsort()[::-1].copy())
    return q[sorted_idx], a[sorted_idx], q_lens[sorted_idx], a_lens[sorted_idx]

######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    matplotlib.use('agg')
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
