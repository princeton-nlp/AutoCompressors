import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

def patch_opt_attn(attn):
    self = attn

    def forward(
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # assert self.is_decoder and not is_cross_attention, "Only decoder layers are implemented in fast_attention"
        assert output_attentions is False, "output_attentions not implemented for fast_attention"
        assert layer_head_mask is None, "layer_head_mask is not supported with fast_attention"
        # assert hidden_states.dtype in [torch.float16, torch.bfloat16], "Only float16 and bfloat16 are supported for fast_attention"

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        query_states = self._shape(query_states, tgt_len, bsz)

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0].to(query_states.dtype)
            value_states = past_key_value[1].to(query_states.dtype)
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0].to(key_states.dtype), key_states], dim=2)
            value_states = torch.cat([past_key_value[1].to(value_states.dtype), value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention: save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # if uni-directional: self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # Note that this does NOT support attention masks other than causal masking
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=True,
            dropout_p=(self.dropout if self.training else 0.0),
        )
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value

    self.forward = forward


def patch_opt(model):
    for layer in model.model.decoder.layers:
        patch_opt_attn(layer.self_attn)
