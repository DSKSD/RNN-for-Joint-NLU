# RNN-for-Joint-NLU

Pytorch implementation of "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling" (https://arxiv.org/pdf/1609.01454.pdf)

<img src="https://github.com/DSKSD/RNN-for-Joint-NLU/raw/master/images/jointnlu0.png"/>

Intent prediction and slot filling are performed in two branches based on Encoder-Decoder model.

## dataset (Atis)

You can get data from <a href="https://github.com/yvchen/JointSLU/tree/master/data ">here</a>


## Requirements

* `Pytorch 0.2`

## Train

`python3 train.py --data_path 'your data path e.g. ./data/atis-2.train.w-intent.iob'`


## Result

<img src="https://github.com/DSKSD/RNN-for-Joint-NLU/raw/master/images/jointnlu1.png"/>
<img src="https://github.com/DSKSD/RNN-for-Joint-NLU/raw/master/images/jointnlu2.png"/>
<img src="https://github.com/DSKSD/RNN-for-Joint-NLU/raw/master/images/jointnlu3.png"/>