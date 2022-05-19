#-*-coding:utf-8-*-
import json
import numpy as np
import warnings
import pickle
import os
from gensim.models import word2vec, KeyedVectors
from allennlp.modules.elmo import Elmo, batch_to_ids

def constrcut_vocab(train_file_path, valid_file_path, test_file_path):
    f_train = open(train_file_path)
    f_valid = open(valid_file_path)
    f_test = open(test_file_path)

    f_list = [f_train, f_valid, f_test]

    vocab_set = set()

    for f in f_list:
        for line in f:
            l_json = json.loads(line)
            text = l_json['sens']

            for word in text:
                vocab_set.add(word)


    word2id = {'pad':0}
    id2word = {0:'pad'}

    for id, word in enumerate(vocab_set):
        word2id[word]  = id
        id2word[id] = word

    #print(word2id)
    #print(id2word)
    return word2id, id2word

def load_vec(fname, word2id, k, isbinary=True):
    model = KeyedVectors.load_word2vec_format(fname, binary=isbinary)
    weights = np.zeros(shape=(word2id.__len__(), k))

    #unknow_weight = np.random.uniform(-0.25, 0.25, k)
    for word in word2id:
        if word not in model.wv.vocab:
            print(word)
            weights[word2id[word]] = np.random.uniform(-0.25, 0.25, k)
        else:
            weights[word2id[word]] = model.wv[word]

    return weights

def transfer_sentence_to_id(fname, word2id):
    res_dict = {}
    senid = []
    sens = []
    frags =[]
    length =[]
    tags = []

    sens_orgs = []

    fr = open(fname, 'r')

    for line in fr:
        r_json = json.loads(line)
        senid.append(int(r_json['senid']))
        frags.append(r_json['fragments'])
        length.append(r_json['length'])
        tags.append(r_json['tags'])
        sen = []
        for i in range(len(r_json['sens'])):
            sen.append(word2id[r_json['sens'][i]])

        sens.append(sen)
        sens_orgs.append(r_json['sens'])


    sen_char_id = batch_to_ids(sens_orgs)
    res_dict = {'senid': senid, 'sens': sens, 'frags':frags, 'length': length, 'tags': tags, 'char_ids':sen_char_id}
    #print(res_dict['senid'])
    #print(res_dict['sens'])
    #print(res_dict['frags'])
    #print(res_dict['length'])
    #print(res_dict['tags'])
    return res_dict




def res_elmo():
    #org_dir = '../my_data/res'
    org_dir = './my_data/res'
    #org_dir = './frag_data/laptop_2014_rand0.6'
    word2id, id2word = constrcut_vocab(os.path.join(org_dir, 'train_data_org.txt'),
                                       os.path.join(org_dir, 'test_data_org.txt'),
                                       os.path.join(org_dir,'valid_data_org.txt'))
    word_embedding_path = "./my_data/w2v_200.bin"
    embed_weights = load_vec(word_embedding_path, word2id, 200)
    #train_data = transfer_sentence_to_id('./temp_data/train_data_1.txt', word2id, 50)
    #valid_data = transfer_sentence_to_id('./temp_data/train_data_1.txt', word2id, 50)
    #test_data = transfer_sentence_to_id('./temp_data/train_data_1.txt', word2id, 50)
    train_data = transfer_sentence_to_id(os.path.join(org_dir, 'train_data_1.txt'), word2id)
    valid_data = transfer_sentence_to_id(os.path.join(org_dir, 'valid_data_1.txt'), word2id)
    test_data = transfer_sentence_to_id(os.path.join(org_dir, 'test_data_1.txt'), word2id)
    pickle.dump([word2id, id2word, embed_weights, train_data, valid_data, test_data],  open('./my_data/predata-res-elmo-mul.pkl', 'wb'))




if __name__ == '__main__':
    res_elmo()



