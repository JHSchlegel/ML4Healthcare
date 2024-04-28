# This script contains the Transformer class and the PositionalEncoding class.

# =========================================================================== #
#                              Packages and Presets                           #
# =========================================================================== #
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def create_padding_mask(seq, pad_token=0.0):
    # mask that is True for padded elements and False for non-padded elements
    mask = (seq == pad_token)  
    return mask

# Transformer for time series classification
class Transformer(nn.Module):
    def __init__(
        self,
        input_size:int = 1,
        model_size:int = 64,
        num_classes:int = 2,
        num_heads:int = 8,
        num_layers:int = 6,
        d_ff:int = 256,
        dropout:float = 0.0,
        transformer_activation: str = "gelu",
        use_padding_mask:bool = False
    ) -> None:
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_size, model_size)
        
        self.pos_encoder = PositionalEncoding(model_size, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
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
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2*model_size, d_ff),
            nn.Hardswish(),
            nn.Linear(d_ff, num_classes)
        )
        
        self.dropout_p = dropout
        self.use_padding_mask = use_padding_mask
        
        if self.dropout_p > 0 and self.use_padding_mask:
            warnings.warn("Cannot use dropout and padding mask at the same time")

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        src = self.embedding(x) 

        src = self.pos_encoder(src)
        
        if self.use_padding_mask:
            src_key_padding_mask = create_padding_mask(x.squeeze(2).T)
        else:
            src_key_padding_mask = None
        

        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )
        # average pooling over sequence:
        avg_pooled = output.mean(dim=0)
        # max pooling over sequence:
        max_pooled = output.max(dim=0).values
        output = torch.cat([avg_pooled, max_pooled], dim=1)
        output = self.decoder(output)
        
        return output
