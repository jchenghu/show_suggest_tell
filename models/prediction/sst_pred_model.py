import torch
from utils.masking import create_pad_mask, create_no_peak_and_pad_mask
from models.prediction.base_pred_module import CaptioningModel

import torch.nn as nn

import math

import numpy as np
import torch.nn.functional as F


from models.prediction.pred_model_layers import AttReduce, \
    ExpNet_EncoderLayer, ExpNet_DecoderLayer

from models.layers import EmbeddingLayer


class SST_ExpNet_Pred(CaptioningModel):
    def __init__(self, k_gram,
                 d_model, N_enc, N_dec, ff, num_heads, num_exp_enc_list, num_exp_dec,
                 output_word2idx, output_idx2word, max_seq_len, drop_args, img_feature_dim=2048, rank=0):
        super().__init__()
        self.k_gram = k_gram
        self.output_word2idx = output_word2idx
        self.output_idx2word = output_idx2word
        self.max_seq_len = max_seq_len

        self.num_exp_dec = num_exp_dec
        self.num_exp_enc_list = num_exp_enc_list

        self.N_enc = N_enc
        self.N_dec = N_dec
        self.d_model = d_model

        self.encoders = nn.ModuleList([ExpNet_EncoderLayer(d_model, ff, num_exp_enc_list, drop_args.enc) for _ in range(N_enc)])
        self.decoders = nn.ModuleList([ExpNet_DecoderLayer(d_model, num_heads, ff, num_exp_dec, drop_args.dec) for _ in range(N_dec)])

        self.sugg_att_reduce = AttReduce(d_model, glimpses=1, out_size=d_model, dropout_perc=drop_args.dec)

        self.input_embedder_dropout = nn.Dropout(drop_args.enc_input)
        self.input_linear = torch.nn.Linear(img_feature_dim, d_model)
        self.vocab_linear = torch.nn.Linear(d_model, len(output_word2idx))
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.out_enc_dropout = nn.Dropout(drop_args.other)
        self.out_dec_dropout = nn.Dropout(drop_args.other)

        self.notconfuse_out_embedder = EmbeddingLayer(len(output_word2idx), d_model, drop_args.dec_input)
        self.k_gram_proj = nn.Linear(k_gram * d_model, d_model)
        self.out_embedder = EmbeddingLayer(len(output_word2idx), d_model, drop_args.dec_input)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        self.enc_reduce_group = nn.Linear(d_model * self.N_enc, d_model)
        self.enc_reduce_norm = nn.LayerNorm(d_model)
        self.dec_reduce_group = nn.Linear(d_model * self.N_dec, d_model)
        self.dec_reduce_norm = nn.LayerNorm(d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.trained_steps = 0
        self.rank = rank

    def forward_enc(self, enc_input, enc_input_num_pads):
        x = self.input_embedder_dropout(self.input_linear(enc_input.float()))

        sum_num_enc = sum(self.num_exp_enc_list)
        pos_x = torch.arange(sum_num_enc).unsqueeze(0).expand(enc_input.size(0), sum_num_enc).to(self.rank)
        pad_mask = create_pad_mask(mask_size=(enc_input.size(0), sum_num_enc, enc_input.size(1)),
                                   pad_row=[0] * enc_input.size(0),
                                   pad_column=enc_input_num_pads,
                                   rank=self.rank)

        x_list = []
        for i in range(self.N_enc):
            x = self.encoders[i](x=x, n_indexes=pos_x, mask=pad_mask)
            x_list.append(x)
        x_list = torch.cat(x_list, dim=-1)
        x = x + self.out_enc_dropout(self.enc_reduce_group(x_list))
        x = self.enc_reduce_norm(x) 
        return x

    def forward_bidir(self,
                      cross_input, enc_input_num_pads,
                      sugg_input,
                      sugg_num_pads
                      ):

        if sugg_input is None:
            bs = len(cross_input)
            return torch.zeros(bs, self.d_model).to(cross_input.device)

        bs, num_ngrams, k_gram = sugg_input.shape

        y = self.notconfuse_out_embedder(sugg_input)
        y = y.reshape(bs, num_ngrams * k_gram, -1)
        # [bs, num_gram_prod_kgram, d_model]

        pos_y = torch.arange(self.k_gram).unsqueeze(0).expand(bs, self.k_gram).to(self.rank)
        pos_y = self.pos_encoder(pos_y)
        # [bs, kgram] -> [bs, kgram, d_model]
        pos_y = pos_y.unsqueeze(1).repeat(1, num_ngrams, 1, 1)
        # [bs, num_gram_prod_kgram // self.k_gram, k_gram, d_model]
        pos_y = pos_y.reshape(bs, num_ngrams * k_gram, self.d_model)
        # [bs, num_gram_prod_kgram, d_model]
        y = y + pos_y
        # [bs, num_gram_prod_kgram, d_model] + [bs, num_gram_prod_kgram, d_model]

        ################################################################################################

        bs, num_ngrams, k_gram = sugg_input.shape
        y = y.reshape(bs, num_ngrams, k_gram, self.d_model)
        y = y.reshape(bs, num_ngrams, k_gram * self.d_model)
        # [bs, num_ngrams, k_gram, d_model]
        y = self.k_gram_proj(y)
        # [bs, num_ngrams, d_model]

        self_mask = create_pad_mask(
            mask_size=(y.size(0), y.size(1), y.size(1)),
            pad_row=torch.tensor(sugg_num_pads),
            pad_column=torch.tensor(sugg_num_pads),
            rank=self.rank)

        sugg_features = self.sugg_att_reduce(y, self_mask)
        return sugg_features

    def forward_dec(self, cross_input, enc_input_num_pads, dec_input, dec_input_num_pads,

                    sugg_input,
                    sugg_num_pads,

                    apply_log_softmax=False):

        no_peak_and_pad_mask = create_no_peak_and_pad_mask(
                                mask_size=(dec_input.size(0), dec_input.size(1), dec_input.size(1)),
                                num_pads=torch.tensor(dec_input_num_pads),
                                rank=self.rank)

        pad_mask = create_pad_mask(mask_size=(dec_input.size(0), dec_input.size(1), cross_input.size(1)),
                                   pad_row=torch.tensor(dec_input_num_pads),
                                   pad_column=torch.tensor(enc_input_num_pads),
                                   rank=self.rank)
        #########################################################################################################

        #sugg_mask_2 = create_pad_mask(
        #    mask_size=(dec_input.size(0), dec_input.size(1), sugg_input.size(1)),
        #    pad_row=torch.tensor(dec_input_num_pads),
        #    pad_column=torch.tensor(sugg_num_pads),
        #    rank=self.rank)

        y = self.out_embedder(dec_input)
        pos_x = torch.arange(self.num_exp_dec).unsqueeze(0).expand(dec_input.size(0), self.num_exp_dec).to(self.rank)
        pos_y = torch.arange(dec_input.size(1)).unsqueeze(0).expand(dec_input.size(0), dec_input.size(1)).to(self.rank)
        y = y + self.pos_encoder(pos_y)
        y_list = []
        for i in range(self.N_dec):
            y = self.decoders[i](x=y,
                                 n_indexes=pos_x,
                                 cross_connection_x=cross_input,
                                 input_attention_mask=no_peak_and_pad_mask,
                                 cross_attention_mask=pad_mask,

                                 sugg_x=sugg_input,
                                 sugg_mask=None
                                 )
            y_list.append(y)
        y_list = torch.cat(y_list, dim=-1)
        y = y + self.out_dec_dropout(self.dec_reduce_group(y_list))
        y = self.dec_reduce_norm(y)

        y = self.vocab_linear(y)

        if apply_log_softmax:
            y = self.log_softmax(y)

        return y
