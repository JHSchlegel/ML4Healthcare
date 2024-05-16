# This script contains the Transformer class and the PositionalEncoding class.
# In case you get the error that torch.backends.mha was not found, run the following command:
# pip install --upgrade torch

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
        # src_key_padding_mask = F._canonical_mask(
        #     mask=src_key_padding_mask,
        #     mask_name="src_key_padding_mask",
        #     other_type=F._none_or_dtype(src_mask),
        #     other_name="src_mask",
        #     target_type=src.dtype,
        # )

        # src_mask = F._canonical_mask(
        #     mask=src_mask,
        #     mask_name="src_mask",
        #     other_type=None,
        #     other_name="",
        #     target_type=src.dtype,
        #     check_other=False,
        # )

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

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ""
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
            src_key_padding_mask is not None or src_mask is not None
        ):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                (x.device.type in _supported_device_type) for x in tensor_args
            ):
                why_not_sparsity_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"{_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(
                    src_mask, src_key_padding_mask, src
                )
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
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


#!!! This function was copied (without any adoptations from our side) from the official pytorch documentation:
#!!! https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


#!!! This function was copied (without any adoptations from our side) from the official pytorch documentation:
#!!! https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


#!!! This function was copied (without any adoptations from our side) from the official pytorch documentation:
#!!! https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


#!!! Most of the forward function in this custom TransformerEncoderWithWeights class was
#!!! adopted from the official pytorch documentation:
#!!! https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
class TransformerEncoderWithWeights(nn.TransformerEncoder):
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        attention_weights = []
        for mod in self.layers:
            output, weights = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )
            attention_weights.append(weights)

        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_weights


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

        self.transformer_encoder = TransformerEncoderWithWeights(
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

        # attention_weights = []
        # output = src
        # for layer in self.transformer_encoder.layers:
        #     output, weights = layer(output, src_key_padding_mask=src_key_padding_mask)
        #     attention_weights.append(weights)
        output, attention_weights = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # average pooling over sequence:
        avg_pooled = output.mean(dim=0)
        # max pooling over sequence:
        max_pooled = output.max(dim=0).values
        output = torch.cat([avg_pooled, max_pooled], dim=1)
        output = self.decoder(output)

        if get_weights:
            return output, attention_weights

        return output
