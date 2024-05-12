# This script contains the Transformer class and the PositionalEncoding class.

# =========================================================================== #
#                              Packages and Presets                           #
# =========================================================================== #
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from icecream import ic


# =========================================================================== #
#               Transformer for Time Series Classification                    #
# =========================================================================== #
# Positional Encoding for Transformer
#!!! copied from https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_10_3_transformer_timeseries.ipynb
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def create_padding_mask(seq, pad_token=0.0):
    # mask that is True for padded elements and False for non-padded elements
    mask = seq == pad_token
    return mask


#!!! based on
#!!! 1. https://discuss.pytorch.org/t/obtain-the-attention-weights-within-transformer-class/165456
#!!! 2. https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayerWithWeights, self).__init__(*args, **kwargs)

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
            is_causal=is_causal,
        )
        return self.dropout1(x), weights

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x_att, weights = self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + x_att
            x = x + self._ff_block(self.norm2(x))
        else:
            x_att, weights = self._sa_block(
                x, src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = self.norm1(x + x_att)
            x = self.norm2(x + self._ff_block(x))

        return x, weights


# Transformer for time series classification
class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        model_size: int = 64,
        num_classes: int = 2,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.0,
        transformer_activation: str = "gelu",
        use_padding_mask: bool = False,
    ) -> None:
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_size, model_size)

        self.pos_encoder = PositionalEncoding(model_size, dropout)

        encoder_layers = TransformerEncoderLayerWithWeights(
            d_model=model_size,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=transformer_activation,
            layer_norm_eps=1e-6,
            batch_first=False,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_size),
            enable_nested_tensor=False,
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * model_size, d_ff),
            nn.Hardswish(),
            nn.Linear(d_ff, num_classes),
        )

        self.dropout_p = dropout
        self.use_padding_mask = use_padding_mask

    def forward(self, x: torch.Tensor, get_weights: bool = False) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        src = self.embedding(x)

        src = self.pos_encoder(src)

        if self.use_padding_mask:
            src_key_padding_mask = create_padding_mask(x.squeeze(2).T)
        else:
            src_key_padding_mask = None

        attention_weights = []
        output = src
        for layer in self.transformer_encoder.layers:
            output, weights = layer(output, src_key_padding_mask=src_key_padding_mask)
            attention_weights.append(weights)

        # average pooling over sequence:
        avg_pooled = output.mean(dim=0)
        # max pooling over sequence:
        max_pooled = output.max(dim=0).values
        output = torch.cat([avg_pooled, max_pooled], dim=1)
        output = self.decoder(output)

        if get_weights:
            return output, attention_weights

        return output
