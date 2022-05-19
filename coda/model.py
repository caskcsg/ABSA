#-*-coding:utf-8-*-

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from allennlp.modules.elmo import Elmo

class AspectModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, input_vocab_size, word_embed_weight, tagset_size, drop=0.5):
        super(AspectModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_vocab_size = input_vocab_size
        self.tagset_size = tagset_size

        self.word_embedding = nn.Embedding(self.input_vocab_size, input_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(np.array(word_embed_weight)))
        self.elmo_dim = 256

        options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/' \
                '2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/' \
                '2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

        self.elmo = Elmo(options_file, weight_file, 2, requires_grad = False, dropout=0)

        self.dropout = nn.Dropout(drop)
        self.encoder = nn.LSTM(input_size=input_dim+self.elmo_dim, hidden_size=hidden_dim, num_layers=2,
                               bidirectional=True, batch_first=True)

        self.att = nn.Linear(hidden_dim*2, hidden_dim*6)
        self.hidden2tag = nn.Linear(hidden_dim*8, tagset_size)

    def init_hidden(self, batch_size):
         h_t_encoder = Variable(torch.zeros(4, batch_size, self.hidden_dim)).cuda()
         c_t_encoder = Variable(torch.zeros(4, batch_size, self.hidden_dim)).cuda()
         return h_t_encoder, c_t_encoder

    def forward(self, sens, frags, length, charids):
        batch_size = sens.size(0)
        temp_charids = charids[:,0:length[0], :]
        elmo_embed = self.elmo(temp_charids)
        embed = self.word_embedding(sens)
        con_embed = torch.cat((embed, elmo_embed['elmo_representations'][1]), dim=2)
        embed = self.dropout(con_embed)

        x_packed = nn.utils.rnn.pack_padded_sequence(embed, lengths=length, batch_first=True)
        output, hidden = self.encoder(x_packed)
        en_output, en_hidden = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)


        total_frag_count = 0
        for item in frags:
            total_frag_count = total_frag_count + len(item)

        seq_length = length[0]
        seq_length_1 = (length[0]).float().cuda()

        word_state = Variable(torch.zeros(total_frag_count, self.hidden_dim*2)).cuda()
        memory_state = Variable(torch.zeros(total_frag_count, seq_length, self.hidden_dim*2)).cuda()
        mask = torch.zeros(total_frag_count, seq_length, 1).byte().cuda()
        l_word_state = Variable(torch.zeros(total_frag_count, self.hidden_dim*2)).cuda()
        r_word_state = Variable(torch.zeros(total_frag_count, self.hidden_dim*2)).cuda()

        ########################
        pos_dis = Variable(torch.zeros(total_frag_count, seq_length)).cuda()
        pos_id = torch.arange(0, seq_length, 1)


        ########################


        k = 0
        for i in range(batch_size):
            for s_frags in frags[i]:
                word_state[k,:] = torch.sum(en_output[i,s_frags[0]:s_frags[1],:], dim=0)
                l_word_state[k, :] = en_output[i, s_frags[0],:]
                r_word_state[k, :] = en_output[i, s_frags[1]-1,:]

                memory_state[k, :, :] = en_output[i, :, :]
                mask[k, s_frags[0]:s_frags[1], :] = 1
                mask[k, length[i]:seq_length, :] = 1
                pos_dis[k, 0:s_frags[0]] = s_frags[0] - pos_id[0:s_frags[0]]
                pos_dis[k, s_frags[0]:s_frags[1]] = seq_length
                pos_dis[k, s_frags[1]:seq_length] = pos_id[s_frags[1]:seq_length]-s_frags[1] + 1

                k = k + 1

        span = torch.cat((l_word_state, word_state, r_word_state), dim=1)

        pos_weight = 1 - torch.div(pos_dis, seq_length_1)
        final = pos_weight.view(total_frag_count, seq_length, 1) * memory_state
        att_matrix = self.att(final)
        mask_att = torch.tanh(torch.bmm(att_matrix, span.view(total_frag_count, -1, 1)))
        mask_att = mask_att.masked_fill(mask, -float(10000))
        att = torch.softmax(mask_att.view(total_frag_count, seq_length), dim = 1)
        mix = torch.bmm(att.view(total_frag_count, -1, seq_length), final)


        mix_state = torch.cat((span, mix.squeeze(1)), dim=1)
        final_out = self.hidden2tag(mix_state)
        label = F.log_softmax(final_out, dim=1)

        return label


if __name__ == '__main__':
    pass



