import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import random
import argparse
import numpy as np
from sklearn.model_selection import KFold
from data import *
from model import Encoder,Decoder
from evalutate import evaluate
from util import set_seed

USE_CUDA = torch.cuda.is_available()


def cross_validate(config, k=10):
    print("Running " + str(k) + "-fold cross-validation. This can take some time.")

    data, word2index, tag2index, intent2index = preprocessing(
            [config.file_path, config.validation_set_file_path, config.test_set_file_path],
            config.max_length)

    if data is None:
        print("Please check your data or its path")
        return

    all_f1_tag_score, all_intent_accuracy = [], []
    kfold = KFold(n_splits=k, shuffle=True, random_state=config.random_seed)
    for i, (train_idxs, eval_idxs) in enumerate(kfold.split(data)):
        print("Repetition " + str(i))
        train_data = [data[i] for i in train_idxs]
        validation_data = [data[i] for i in eval_idxs]
        f1_tag_score, intent_accuracy = train(
                config, train_data, validation_data,
                word2index=word2index, tag2index=tag2index, 
                intent2index=intent2index
        )

        all_f1_tag_score.append(f1_tag_score)
        all_intent_accuracy.append(intent_accuracy)
    print("===================================================")
    print(str(k) + "-fold cross validation complete. Average scores:")
    print("Tag F1 score: ", np.mean(all_f1_tag_score), 
          ", intent accuracy: ", np.mean(all_intent_accuracy))
    print("===================================================")

def train(config, train_data=None, validation_data=None, test_data=None, 
          word2index=None, tag2index=None, intent2index=None):
    assert (train_data is None and validation_data is None and word2index is
            None and tag2index is None and intent2index is None 
            or
            not (train_data is None or validation_data is None or word2index is
            None or tag2index is None or intent2index is None))

    set_seed(config.random_seed)

    if train_data is None and validation_data is None:
        train_data, word2index, tag2index, intent2index = preprocessing(config.file_path, config.max_length)
        validation_data, _, _, _ = preprocessing(config.validation_set_file_path, config.max_length, word2index, tag2index, intent2index)
        test_data, _, _, _ = preprocessing(config.test_set_file_path, config.max_length, word2index, tag2index, intent2index)
        if train_data is None or validation_data is None or test_data is None:
            print("Please check your data or its path")
            return

    encoder = Encoder(len(word2index),config.embedding_size,config.hidden_size)
    decoder = Decoder(len(tag2index),len(intent2index),len(tag2index)//3,config.hidden_size*2)
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder.init_weights()
    decoder.init_weights()

    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()
    enc_optim= optim.Adam(encoder.parameters(), lr=config.learning_rate)
    dec_optim = optim.Adam(decoder.parameters(),lr=config.learning_rate)

    for step in range(config.step_size):
        losses=[]
        for i, batch in enumerate(getBatch(config.batch_size,train_data)):
            x,y_1,y_2 = zip(*batch)
            x = torch.cat(x).cuda() if USE_CUDA else torch.cat(x)
            tag_target = torch.cat(y_1).cuda() if USE_CUDA else torch.cat(y_1)
            intent_target = torch.cat(y_2).cuda() if USE_CUDA else torch.cat(y_2)

            x_mask = torch.cat([torch.tensor(tuple(map(lambda s: s ==0, t.data)), dtype=torch.bool) for t in x])
            x_mask = x_mask.view(config.batch_size,-1)

            encoder.zero_grad()
            decoder.zero_grad()

            output, hidden_c = encoder(x,x_mask)
            start_decode = Variable(torch.LongTensor([[word2index['<SOS>']]*config.batch_size])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<SOS>']]*config.batch_size])).transpose(1,0)

            tag_score, intent_score = decoder(start_decode,hidden_c,output,x_mask)

            loss_1 = loss_function_1(tag_score,tag_target.view(-1))
            loss_2 = loss_function_2(intent_score,intent_target)

            loss = loss_1+loss_2
            losses.append(loss.data.cpu().numpy().item() if USE_CUDA else loss.data.numpy().item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)

            enc_optim.step()
            dec_optim.step()
            if i % 100==0:
                eval_losses, f1_tag_score, intent_accuracy = evaluate(encoder, decoder, word2index, validation_data, config.batch_size)
                print("Step", step, " epoch", i, ". train_loss: ",
                      np.mean(losses), "eval_loss: ", np.mean(eval_losses), ", tag F1 score: ",
                      f1_tag_score, ", intent accuracy: ", intent_accuracy)
                losses=[]

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    # note: if running cross-validation, the most recently saved model will
    # simply be overwritten by the current one.
    torch.save(decoder.state_dict(),os.path.join(config.model_dir,'jointnlu-decoder.pkl'))
    torch.save(encoder.state_dict(),os.path.join(config.model_dir, 'jointnlu-encoder.pkl'))
    print("Train Complete!")

    if test_data is None:
        test_data = validation_data

    print("Evaluating on test data (or validation data if test data is not available).")
    _, test_f1_tag_score, test_intent_accuracy = evaluate(encoder, decoder, word2index, test_data, config.batch_size)
    print("Tag F1 score: ", test_f1_tag_score, ", intent accuracy: ",
          test_intent_accuracy, "\n\n")
    
    return test_f1_tag_score, test_intent_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/atis-2.train.w-intent.iob' ,
                        help='path to the train data')
    parser.add_argument('--validation_set_file_path', type=str, default='./data/atis-2.dev.w-intent.iob' ,
                        help='path to the validation data')
    parser.add_argument('--test_set_file_path', type=str, default='./data/atis.test.w-intent.iob' ,
                        help='path to the test data')
    parser.add_argument('--model_dir', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--cross_validate', action='store_true', default=False,
                        help='Set this flag to perform 10-fold cross-validation.')

    # Model parameters
    parser.add_argument('--max_length', type=int , default=60 ,
                        help='max sequence length')
    parser.add_argument('--embedding_size', type=int , default=64 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=64 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')

    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--random_seed', type=int, default=1337)
    config = parser.parse_args()

    if config.cross_validate:
        cross_validate(config, k=10)
    else:
        train(config)
