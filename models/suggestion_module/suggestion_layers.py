
import torch
import torch.nn as nn
import math

import numpy as np
import torch.nn.functional as F


from models.layers import MultiHeadAttention, FeedForward


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, rank=0):
        super().__init__()
        assert d_model % 2 == 0, "d_model is not even, even number suggested"
        self.d_model = d_model
        self.pe = torch.zeros(max_seq_len, d_model).to(rank)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe.data[pos, i] = math.sin(pos / (10000.0 ** ((2.0 * i) / d_model)))
                self.pe.data[pos, i + 1] = math.cos(pos / (10000.0 ** ((2.0 * i) / d_model)))
        self.pe.data = self.pe.data.unsqueeze(0)

    # x shape [ batch_size, seq_len, d_model]
    def forward(self, x):
        seq_len = x.shape[1]
        # we apply this to each row of the batch, it's automatically
        # broadcasted all along batches thanks to the pe = pe.unsqueeze(0)
        return self.pe.data[0, :seq_len]

    def apply_according_to_t(self, batch_t):
        # [bs]
        output_pe = []
        for t in batch_t:
            output_pe.append(self.pe.data[0, t])
        output_pe = torch.stack(output_pe, dim=0)
        return output_pe


class DecoderTransfLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout_perc,
                 max_diffusion_steps,

                 eps=1e-9):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.mha_1 = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.mha_2 = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, cross_connection_x, input_attention_mask, cross_attention_mask):
        # Pre-LayerNorm
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.mha_1(q=x2, k=x2, v=x2,
                                          mask=input_attention_mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.mha_2(q=x2, k=cross_connection_x, v=cross_connection_x,
                                          mask=cross_attention_mask))

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class EncoderTransfLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout_perc, eps=1e-9):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)

        self.mha_1 = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.mha_1(q=x2, k=x2, v=x2,
                                          mask=mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
