import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import torch.distributions as dists

from models.suggestion_module.suggestion_layers import \
    EncoderTransfLayer, DecoderTransfLayer, PositionalEncoder
from models.layers import EmbeddingLayer, MultiHeadAttention

from utils.masking import create_pad_mask


from models.suggestion_module.base_sugg_module import BaseSuggestionModule

class SST_Sugg_Module(BaseSuggestionModule):
    def __init__(self,
                 k_gram,
                 d_model, ff, num_heads,
                 num_layers, drop_args,

                 output_word2idx, output_idx2word,

                 max_seq_len,
                 img_feature_dim,

                 num_diffusion_steps,

                 rank
                 ):
        super().__init__()

        self.k_gram = k_gram

        self.output_word2idx = output_word2idx
        self.output_idx2word = output_idx2word
        self.max_seq_len = max_seq_len

        self.num_diffusion_steps = num_diffusion_steps

        self.time_positional_encoder = PositionalEncoder(d_model, max_seq_len, rank)
        self.t_proj_1 = nn.Linear(d_model, d_model * 2)
        self.t_proj_2 = nn.Linear(d_model * 2, d_model)

        self.length_pred_mha = MultiHeadAttention(d_model, num_heads, dropout_perc=drop_args.enc)
        self.length_pred_norm = nn.LayerNorm(d_model)
        self.length_pred_proj = nn.Linear(d_model, 10000)
        # 10,000 it's just an arbitrary big number

        self.mask_idx = self.output_word2idx['MASK']

        self.d_model = d_model
        self.num_layers = num_layers

        self.encoders = nn.ModuleList(
            [EncoderTransfLayer(d_model, ff, num_heads, drop_args.enc) for _ in range(num_layers)])
        self.decoders = nn.ModuleList(
            [DecoderTransfLayer(d_model, ff, num_heads, drop_args.dec,
                                max_diffusion_steps=num_diffusion_steps) for _ in range(num_layers)])

        self.input_embedder_dropout = nn.Dropout(drop_args.enc_input)
        self.input_linear = torch.nn.Linear(img_feature_dim, d_model)
        self.vocab_linear = torch.nn.Linear(d_model, len(output_word2idx))
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.out_enc_dropout = nn.Dropout(drop_args.other)
        self.out_dec_dropout = nn.Dropout(drop_args.other)

        self.ngram_embedder = EmbeddingLayer(10000, d_model, drop_args.dec_input)
        # same with max length pred

        self.out_embedder = EmbeddingLayer(len(output_word2idx), d_model, drop_args.dec_input)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.trained_steps = 0
        self.rank = rank

    def forward_enc(self, enc_input, enc_input_num_pads):
        x = self.input_embedder_dropout(self.input_linear(enc_input.float()))

        pad_mask = create_pad_mask(mask_size=(enc_input.size(0), enc_input.size(1), enc_input.size(1)),
                                   pad_row=enc_input_num_pads,
                                   pad_column=enc_input_num_pads,
                                   rank=self.rank)

        for i in range(self.num_layers):
            x = self.encoders[i](x=x, mask=pad_mask)
        return x

    def forward_length(self, enc_input, enc_input_num_pads):
        pad_mask = create_pad_mask(mask_size=(enc_input.size(0), enc_input.size(1), enc_input.size(1)),
                                   pad_row=enc_input_num_pads,
                                   pad_column=enc_input_num_pads,
                                   rank=self.rank)
        x2 = self.length_pred_norm(enc_input)
        x = enc_input + self.length_pred_mha(q=x2, k=x2, v=x2, mask=pad_mask)
        x = self.length_pred_proj(x)
        length_pred = x.mean(1)
        return length_pred


    def forward_dec(self, cross_input, enc_input_num_pads, dec_input,
                    dec_input_num_pads,
                    diffusion_t,
                    apply_log_softmax=False):
        assert (len(dec_input.shape) == 2), " I expect the UNPACKED dec input."
        bs, num_gram_prod_kgram = dec_input.shape

        cross_pad_mask = create_pad_mask(mask_size=(bs, num_gram_prod_kgram, cross_input.size(1)),
                                         pad_row=torch.tensor(dec_input_num_pads),
                                         pad_column=torch.tensor(enc_input_num_pads),
                                         rank=self.rank)

        diffusion_pad_mask = create_pad_mask(mask_size=(bs, num_gram_prod_kgram, num_gram_prod_kgram),
                                             pad_row=torch.tensor(dec_input_num_pads),
                                             pad_column=torch.tensor(dec_input_num_pads),
                                             rank=self.rank)

        pos_q = torch.arange(num_gram_prod_kgram // self.k_gram).unsqueeze(0).repeat(bs, 1).to(self.rank)
        # [bs, num_gram_prodkgram // self.k_gram]
        pos_q = self.ngram_embedder(pos_q)
        # [bs, num_gram_prodkgram // self.k_gram, d_model]
        pos_q = pos_q.unsqueeze(2).repeat(1, 1, self.k_gram, 1)
        # [bs, num_gram_prodkgram // self.k_gram, self.k_gram, d_model]
        pos_q = pos_q.reshape(bs, num_gram_prod_kgram, -1)
        # [bs, num_gram_prodkgram, d_model]

        y = self.out_embedder(dec_input)
        # [bs, num_gram_prod_kgram, d_model]

        pos_y = torch.arange(self.k_gram).unsqueeze(0).expand(bs, self.k_gram).to(self.rank)
        pos_y = self.pos_encoder(pos_y)
        # [bs, kgram] -> [bs, kgram, d_model]
        pos_y = pos_y.unsqueeze(1).repeat(1, num_gram_prod_kgram // self.k_gram, 1, 1)
        # [bs, num_gram_prod_kgram // self.k_gram, k_gram, d_model]
        pos_y = pos_y.reshape(bs, num_gram_prod_kgram, self.d_model)
        # [bs, num_gram_prod_kgram, d_model]
        y = y + pos_y + pos_q
        # [bs, num_gram_prod_kgram, d_model] + [bs, num_gram_prod_kgram, d_model]

        t_embed = self.time_positional_encoder.apply_according_to_t(diffusion_t)
        t_embed = self.t_proj_2(F.silu(self.t_proj_1(t_embed)))
        t_embed = t_embed.view(y.shape[0], 1, y.shape[-1]).repeat(1, y.shape[1], 1)
        y += t_embed

        for i in range(self.num_layers):
            y = self.decoders[i](x=y,
                                 cross_connection_x=cross_input,
                                 input_attention_mask=diffusion_pad_mask,
                                 cross_attention_mask=cross_pad_mask)

        y = self.vocab_linear(y)

        if apply_log_softmax:
            y = self.log_softmax(y)

        return y

    def time_sample(self, batch_size, rank):
        w = np.ones(self.num_diffusion_steps, dtype=float)  # np.float)
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(rank)
        weights_np = 1 / p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(rank)
        return indices, weights

    def q_sample(self, x_0, t, non_special_sym_mask):
        # samples q(x_t | x_0), randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()
        mask = torch.rand_like(x_0.float()) < ((t.float().unsqueeze(-1) + 1) / self.num_diffusion_steps)
        x_t[mask] = self.mask_idx

        mask = mask & non_special_sym_mask
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def diffuse_absorbing(self, cross_input, enc_input_num_pads, dec_input, dec_input_num_pads,
                          apply_log_softmax=False):

        bs, num_ngrams, k_gram = dec_input.shape

        length_pred = self.forward_length(cross_input, enc_input_num_pads)
        length_target = [dec_input.size(1) - num_pad for num_pad in dec_input_num_pads]
        length_target = torch.tensor(length_target, dtype=torch.long).to(self.rank)

        non_special_sym_mask = dec_input.ne(self.output_word2idx['PAD'])

        # sample t, and q_sample (x_t|x_0)
        all_steps_diffuse_dict = []
        for t in range(self.num_diffusion_steps):

            w = torch.ones((self.num_diffusion_steps)).float().to(dec_input.device)  # np.float)
            p = w / w.sum()
            batch_size = dec_input.size(0)
            t_vector = torch.tensor([t for _ in range(batch_size)]).long().to(dec_input.device)
            weights = 1 / p[t_vector]

            #
            UNPACK_dec_input = dec_input.reshape(bs, num_ngrams * k_gram)
            UNPACK_non_special_sym_mask = non_special_sym_mask.reshape(bs, num_ngrams * k_gram)
            UNPACK_x_t, UNPACK_x_0_ignore, UNPACK_mask = self.q_sample(
                x_0=UNPACK_dec_input, t=t_vector,
                non_special_sym_mask=UNPACK_non_special_sym_mask)
            x_0_ignore = UNPACK_x_0_ignore.reshape(bs, num_ngrams, k_gram)
            mask = UNPACK_mask.reshape(bs, num_ngrams, k_gram)
            x_t = UNPACK_x_t.reshape(bs, num_ngrams, k_gram)

            # since we doubled the num_ngrams we multiply also num_pads
            UNPACK_dec_input_num_pads = [ num_ngram * self.k_gram for num_ngram in dec_input_num_pads]
            UNPACK_decoder_output = self.forward_dec(
                cross_input=cross_input, enc_input_num_pads=enc_input_num_pads,
                dec_input=UNPACK_x_t, dec_input_num_pads=UNPACK_dec_input_num_pads,
                diffusion_t=t_vector,
                apply_log_softmax=apply_log_softmax
            )
            decoder_output = UNPACK_decoder_output.reshape(bs, num_ngrams, k_gram, -1)

            diffuse_dict = {
                'x_t': x_t, 't': t_vector, 'x_0_ignore': x_0_ignore,
                'masks': mask, 'weights': weights,
                'decoder_output': decoder_output,
                'length_pred': length_pred, 'length_target': length_target}
            all_steps_diffuse_dict.append(diffuse_dict)

        return all_steps_diffuse_dict
