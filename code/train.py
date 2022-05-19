import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import time
import shutil


from torch.autograd import Variable
from torch.utils.data import *

from model import *
import argparse


def batch_gen(sens, senid, frags, length, tags, charids, batch_size, shuffle=True):
    len_data = len(senid)

    if shuffle:
        indices = torch.randperm(len_data).long().cuda()
    else:
        indices = torch.arange(len_data).long().cuda()


    start_index = 0
    while start_index < len_data:
        #获取一个batch的index
        end_index = start_index + batch_size
        if end_index > len_data:
            end_index = len_data
        temp_sens = [sens[i] for i in indices[start_index:end_index]]
        temp_frags = [frags[i] for i in indices[start_index:end_index]]
        temp_tags = [tags[i] for i in indices[start_index:end_index]]
        temp_senid = torch.index_select(senid, 0, indices[start_index:end_index])
        temp_length = torch.index_select(length, 0, indices[start_index:end_index])
        temp_charid = torch.index_select(charids, 0, indices[start_index:end_index])

        res_length, l_index = torch.sort(temp_length, 0, True)
        res_senid = torch.index_select(temp_senid, 0, l_index)
        res_charid = torch.index_select(temp_charid, 0, l_index)

        batch = end_index-start_index
        temp_res_sens = [temp_sens[i] for i in l_index]
        res_frags = [temp_frags[i] for i in l_index]
        res_sens = torch.zeros(batch, res_length[0]).long().cuda()

        res_tags = []
        for i in l_index:
            res_tags = res_tags + temp_tags[i]

        tensor_tags = torch.from_numpy(np.array(res_tags)).cuda()

        for i in range((batch)):
            for j in range(len(temp_res_sens[i])):
                res_sens[i][j] = temp_res_sens[i][j]

        start_index = end_index

        yield res_sens, res_frags, res_length, tensor_tags, res_senid, res_charid


def get_numbers(true_labels, predict_value):

    tag_len = true_labels.size()[0]
    v, pos = torch.max(predict_value, dim = 1)
    predict_tag = torch.zeros(tag_len)

    predict_number = 0
    true_label_number = 0
    cor_number = 0
    for i in range(tag_len):
        if pos[i] != 0:
            predict_number += 1
        if true_labels[i] != 0:
            true_label_number += 1
        if pos[i] != 0 and pos[i] == true_labels[i]:
            cor_number += 1

    return predict_number, true_label_number, cor_number

def get_numbers_boundary_predict(true_labels, predict_value):

    tag_len = true_labels.size()[0]
    v, pos = torch.max(predict_value, dim = 1)
    predict_tag = torch.zeros(tag_len)

    predict_number = 0
    true_label_number = 0
    cor_number = 0

    l_true_number = 0
    #boundary
    for i in range(tag_len):
        if pos[i] != 0:
            predict_number += 1
        if true_labels[i] != 0:
            true_label_number += 1
        if pos[i] != 0 and true_labels[i] != 0 :
            cor_number += 1
            if pos[i] == true_labels[i]:
                l_true_number += 1

    return predict_number, true_label_number, cor_number, l_true_number

def score(true_count, pre_count, correct_count):
    p = float(correct_count)/pre_count if pre_count!=0 else 0
    r = float(correct_count)/true_count if true_count!=0 else 0
    f1 =(2*p*r)/(p+r) if p!=0  else 0

    return p, r, f1




def run(lr, batch, hidden, dropout, bias, fname):
    input_dim = 200
    hidden_dim = hidden
    batch_size = batch

    print('lr:', lr)
    print('batch:', batch)
    print('hidden:', hidden)
    print('dropout:', dropout)
    print('bias:', bias)


    word2id, id2word, embed_weights, train_data, valid_data, test_data = pickle.load(open(fname, 'rb'))
    model = AspectModel(input_dim, hidden_dim, len(word2id), embed_weights, 4, drop=dropout).cuda()
    opt = optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-06)
    loss_func = nn.NLLLoss().cuda()


    ############################################
    train_senid_tensor = torch.from_numpy(np.array(train_data['senid'])).cuda()
    train_length_tensor = torch.from_numpy(np.array(train_data['length'])).cuda()
    train_charid_tensor = train_data['char_ids'].cuda()

    valid_senid_tensor = torch.from_numpy(np.array(valid_data['senid'])).cuda()
    valid_length_tensor = torch.from_numpy(np.array(valid_data['length'])).cuda()
    valid_charid_tensor = valid_data['char_ids'].cuda()


    test_senid_tensor = torch.from_numpy(np.array(test_data['senid'])).cuda()
    test_length_tensor = torch.from_numpy(np.array(test_data['length'])).cuda()
    test_charid_tensor = test_data['char_ids'].cuda()
    ############################################


    pre_f1 = 0
    for i in range(20):
        print("epoch--------------------" + str(i))

        for batch in batch_gen(train_data['sens'], train_senid_tensor, train_data['frags'],
                               train_length_tensor, train_data['tags'], train_charid_tensor, batch_size, True):
            b_t_sens, b_t_frags, b_t_length, b_t_tags, b_t_senid, b_t_charid = batch
            model.train()
            model.zero_grad()
            pre_y = model(b_t_sens, b_t_frags, b_t_length, b_t_charid)
            loss = loss_func(pre_y, Variable((b_t_tags).cuda()))
            print('loss,',loss)
            loss.backward()
            opt.step()

        t_predict_number = 0
        t_true_number = 0
        t_correct_number = 0
        model.eval()


        for valid_batch in batch_gen(valid_data['sens'], valid_senid_tensor, valid_data['frags'],
                               valid_length_tensor, valid_data['tags'], valid_charid_tensor, batch_size, True):
            b_v_sens, b_v_frags, b_v_length, b_v_tags, b_v_senid, b_v_charid = valid_batch

            valid_y = model(b_v_sens, b_v_frags, b_v_length, b_v_charid)

            predict_number, true_label_number, cor_number = get_numbers(b_v_tags, valid_y)
            t_predict_number += predict_number
            t_true_number += true_label_number
            t_correct_number += cor_number


        v_p, v_r, v_f1 = score(t_true_number, t_predict_number, t_correct_number)
        print('total_number:%s,%s,%s:'%(t_predict_number, t_true_number, t_correct_number))
        print('----p,r,f1:----%s,%s,%s'%(v_p, v_r, v_f1))
        if v_f1 > pre_f1:
            print('----------------------test------------------------')
            pre_count = 0
            #laptop 634
            true_count = 2287
            correct_count = 0
            pre_true_count = 0

            #boundary
            b_true_count = 0
            b_pre_count = 0
            b_correct_count = 0
            l_correct_count = 0
            for test_batch in batch_gen(test_data['sens'], test_senid_tensor, test_data['frags'],
                               test_length_tensor, test_data['tags'], test_charid_tensor, batch_size, True):
                b_test_sens, b_test_frags, b_test_length, b_test_tags, b_test_senid, b_test_charid = test_batch

                test_y = model(b_test_sens, b_test_frags, b_test_length, b_test_charid)

                test_predict_number, test_true_label_number, test_cor_number = get_numbers(b_test_tags, test_y)
                pre_count += test_predict_number
                pre_true_count += test_true_label_number
                correct_count += test_cor_number
                #boundary and label
                temp_b_predict_number, temp_b_true_number, temp_b_cor_number, \
                        temp_l_true_number = get_numbers_boundary_predict(b_test_tags, test_y)


                b_true_count += temp_b_true_number
                b_pre_count += temp_b_predict_number
                b_correct_count += temp_b_cor_number
                l_correct_count += temp_l_true_number


            t_p, t_r, t_f1 = score(true_count, pre_count, correct_count)

            print('test_total_number:%s,%s,%s,%s:'%(pre_count, true_count, correct_count,pre_true_count))
            print('----test-------p,r,f1:----%s,%s,%s'%(t_p,t_r,t_f1))

            #boundary
            #b_p, b_r, b_f1 = score(true_count, b_pre_count, b_correct_count)
            #print('boundary_total_number:%s,%s,%s,%s:'%(b_pre_count, true_count, b_correct_count, b_true_count))
            #print('----boundary test-------p,r,f1:----%s,%s,%s'%(b_p, b_r, b_f1))

            #sentiment
            #acc = float(l_correct_count) / b_correct_count if b_correct_count !=0 else 0
            #print('sentiment_total_number:%s,%s:'%(l_correct_count, b_correct_count))
            #print('----sentiment_test-------acc:----%s'%(acc))

            if v_f1 > pre_f1:
                print('best----test-------p,r,f1:----%s,%s,%s'%(t_p,t_r,t_f1))
                pre_f1 = v_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--bias', type=float, default=1)
	parser.add_argument('--fname', default='./mydata/predata-res-elmo.pkl')

	

    args = parser.parse_args()
    run(args.lr, args.batch, args.hidden, args.drop, args.bias, fname)



