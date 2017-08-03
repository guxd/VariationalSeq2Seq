# VariationalSeq2Seq
A pytorch implementation of the paper "Latent Variable Dialogue Models and their Diversity"  https://arxiv.org/abs/1702.05962
It introduces a Variational Sequence to Sequence Model to generate diverse dialogues. 


## Requirements
* [Python 2.7](https://www.continuum.io/downloads)
* [Pytorch](http://pytorch.org/)

## Usage 

#### 1. Prepare dataset
Extract `train.txt` and `valid.txt` from opensubtitle(https://s3.amazonaws.com/opennmt-trainingdata/opensub_qa_en.tgz) to data/
```bash
$ python prepare_data_opensub.py
```

#### 2. Train and Test
```bash
$ python main.py
```




