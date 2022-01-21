# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math

class MHA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):


        super().__init__()    
        
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.h = nhead
                                            
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k ,v):
        r"""
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - Outputs:
          - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
                          E is the embedding dimension.
          - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
                                  L is the target sequence length, S is the source sequence length.
        """
        L = len(q)
        S = len(k)

        k = self.k_linear(k).view(S, -1, self.h, self.d_k)
        q = self.q_linear(q).view(L, -1, self.h, self.d_k)
        v = self.v_linear(v).view(S, -1, self.h, self.d_k).permute(1, 2, 0, 3)
        
        if L == 1:
            scores = (q * k).sum(3).permute(1,2,0).view(-1, self.h, L, S) / math.sqrt(self.d_k)
        else:
            k = k.permute(1, 2, 0, 3)
            q = q.permute(1, 2, 0, 3)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
         
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        concat = output.permute(2, 0, 1, 3).contiguous().view(L, -1, self.d_model)
        output = self.out(concat)

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output



        
class TransformerEncoderLayer3D(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, out_dim=-1, dropout=0.1,
                 activation="relu", normalize_before=True, linformer=False, skip=False):
        super().__init__()

        self.linformer = False
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        if out_dim == -1: 
            out_dim = d_model
        self.linear2 = nn.Linear(dim_feedforward, out_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.skip = skip

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):

        return tensor if pos is None else tensor + pos.repeat(1, tensor.shape[1]//pos.shape[1], 1)


    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):

        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        q = k = src2

        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        return src2

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        return self.forward_pre(src, src_mask, src_key_padding_mask, pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
