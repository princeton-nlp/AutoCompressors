import logging
import os
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import OPTForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

import os

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    """Keep track of token constitution of current input sequence"""
    softprompt_length: int = 0
    past_softprompts_length: int = 0
    summary_length: int = 0

    def reset(self):
        self.softprompt_length = 0
        self.past_softprompts_length = 0
        self.summary_length = 0


class OPTLearnedPositionalEmbeddingWithPadding(nn.Embedding):
    """Overwrite the default OPTLearnedPositionalEmbedding to disable position on summary tokens"""

    def __init__(self, num_embeddings: int, embedding_dim: int, summary_config: Optional[SummaryConfig] = None):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        super().__init__(num_embeddings + 2, embedding_dim, padding_idx=1)

        self.summary_config = summary_config if summary_config is not None else SummaryConfig()

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        attention_mask = attention_mask.long()
        bsz = attention_mask.size(0)

        left_placeholder = torch.ones(bsz, self.summary_config.softprompt_length, dtype=torch.long, device=attention_mask.device) # <pad> -> zero vector
        right_placeholder = torch.ones(bsz, self.summary_config.summary_length, dtype=torch.long, device=attention_mask.device) # <pad> -> zero vector

        total_softprompt_length = self.summary_config.past_softprompts_length + self.summary_config.softprompt_length
        attention_mask = attention_mask[:, total_softprompt_length:attention_mask.size(1)-self.summary_config.summary_length]

        positions = attention_mask.cumsum(dim=1) * attention_mask + 1
        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length - self.summary_config.past_softprompts_length :]
        positions = torch.cat([left_placeholder, positions, right_placeholder], dim=1)

        return super().forward(positions)


@dataclass
class CausalACOutputWithPast(CausalLMOutputWithPast):
    softprompt: Optional[torch.FloatTensor]= None
    softprompt_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class AutoCompressorModel(OPTForCausalLM):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        assert hasattr(self.config, 'summary_length'), "Compressor requires a summary_length config parameter"

        self.summary_config = SummaryConfig()

        self.model.decoder.embed_positions = OPTLearnedPositionalEmbeddingWithPadding(
            config.max_position_embeddings, config.hidden_size, summary_config=self.summary_config
        )

        if config.summary_length > 0:
            self.embed_summary = nn.Embedding(config.summary_length, config.word_embed_proj_dim)

            input_embeds = self.get_input_embeddings()
            self.embed_summary.weight.data[:,:] = (
                input_embeds.weight[config.eos_token_id]
            )

        # Initialize weights and apply final processing
        self.post_init()

    def forward_segment(
        self,
        softprompt: torch.FloatTensor,
        segment_embeds: torch.FloatTensor,
        placeholder_embeds: torch.FloatTensor,
        segment_attention_mask: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        output_hidden_states: bool,
        use_cache: bool,
        output_attentions: bool,
        segment_gradient_checkpointing: bool,
    ):
        # Soon: correct treatment of softprompt past key values
        softprompt_past_key_values_length = 0
        softprompt_past_key_values = None

        segment_embeds = torch.cat([softprompt, segment_embeds, placeholder_embeds], dim=1)

        bsz = segment_embeds.size(0)
        softprompt_length = softprompt.size(1)
        summary_length = placeholder_embeds.size(1)

        device = segment_embeds.device
        attn_dtype = segment_attention_mask.dtype
        segment_attention_mask = torch.cat([
            torch.ones(bsz, softprompt_length, device=device, dtype=attn_dtype),
            segment_attention_mask,
            torch.ones(bsz, summary_length, device=device, dtype=attn_dtype)
        ], dim=1)

        if past_key_values is None:
            segment_past_key_values = softprompt_past_key_values
        elif softprompt_past_key_values is not None:
            segment_past_key_values = tuple(
                tuple(
                    torch.cat([tensor_softprompt, tensor], dim=2)
                    for tensor_softprompt, tensor in zip(tensors_softprompt, tensors)
                )
                for tensors_softprompt, tensors in zip(softprompt_past_key_values, past_key_values)
            )

        def decoder(segment_embeds,
                    segment_attention_mask,
                    segment_past_key_values,
                    softprompt_length,
                    softprompt_past_key_values_length,
                    summary_length):
            self.summary_config.softprompt_length = softprompt_length
            self.summary_config.past_softprompts_length = softprompt_past_key_values_length
            self.summary_config.summary_length = summary_length

            return self.model.decoder(
                inputs_embeds=segment_embeds,
                attention_mask=segment_attention_mask,
                past_key_values=segment_past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,)

        if segment_gradient_checkpointing:
            outputs = torch.utils.checkpoint.checkpoint(
                decoder, segment_embeds, segment_attention_mask, segment_past_key_values,
                softprompt_length, softprompt_past_key_values_length, summary_length,
                use_reentrant=False)
        else:
            outputs = decoder(
                segment_embeds, segment_attention_mask, segment_past_key_values,
                softprompt_length, softprompt_past_key_values_length, summary_length)

        return outputs


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        segment_lengths: Optional[Union[List[int], int]] = None,
        softprompt: Optional[torch.FloatTensor] = None,
        output_softprompt: Optional[bool] = None,
        softprompt_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if past_key_values is not None or softprompt_past_key_values is not None:
            raise ValueError("Support for past key values are experimental and currently not supported")
        past_key_values_length = 0 # until further change

        if head_mask is not None:
            raise ValueError("Compressor does not support head_mask")
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Compressor does not support both input_ids and input_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        segment_lengths = segment_lengths if segment_lengths is not None else input_ids.size(1)

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.size(0),
                                        inputs_embeds.size(1) + past_key_values_length,
                                        dtype=torch.long, device=inputs_embeds.device)

        if self.config.summary_length > 0:
            placeholder_ids = torch.arange(self.config.summary_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.size(0), -1)
            placeholder_embeds = self.embed_summary(placeholder_ids)
        else:
            placeholder_embeds = inputs_embeds[:,:0]

        inputs_embeds_list = torch.split(inputs_embeds, segment_lengths, dim=1)
        attention_mask_list = torch.split(attention_mask[:, past_key_values_length:], segment_lengths, dim=1)

        placeholder_inputs_embeds_list = [placeholder_embeds] * (len(inputs_embeds_list) - 1) + [placeholder_embeds if output_softprompt else placeholder_embeds[:,:0,:]]

        last_hidden_state_list = []
        output_attentions_list = []
        output_hidden_states_list = []

        if softprompt is None:
            softprompt = inputs_embeds[:,:0,:]

        for step, (segment_embeds, placeholder_embeds, segment_attention_mask) in enumerate(zip(
                inputs_embeds_list, placeholder_inputs_embeds_list, attention_mask_list)):

            is_last_step = step == len(inputs_embeds_list) - 1
            segment_gradient_checkpointing = (
                getattr(self.config, "segment_gradient_checkpointing", False) and
                self.training and not is_last_step)

            outputs = self.forward_segment(
                softprompt, segment_embeds, placeholder_embeds, segment_attention_mask,
                past_key_values, output_hidden_states, use_cache, output_attentions,
                segment_gradient_checkpointing)

            softprompt_length = self.summary_config.softprompt_length
            summary_length = self.summary_config.summary_length

            total_length = outputs.last_hidden_state.size(1)
            segment_last_hiddens = (
                outputs.last_hidden_state[:,softprompt_length:total_length-summary_length]
            )
            last_hidden_state_list.append(segment_last_hiddens)

            new_softprompt = outputs.last_hidden_state[:,total_length-summary_length:]
            if self.config.accumulate_summary:
                softprompt = torch.cat([softprompt, new_softprompt], dim=1)
                if outputs.past_key_values is not None:
                    softprompt_past_key_values = tuple(
                        tuple(tensor[:, :, :softprompt_length] for tensor in tensors)
                        for tensors in outputs.past_key_values
                    )
                    # softprompt_past_key_values_length = softprompt_past_key_values[0][0].shape[-2]
            else:
                softprompt = new_softprompt
                softprompt_past_key_values = None
                # softprompt_past_key_values_length = 0

            output_attentions_list.append(outputs.attentions)
            output_hidden_states_list.append(outputs.hidden_states)

            # only use past_key_value in first segment
            past_key_values = None
            past_key_values_length = 0

        # Reset placeholder positions
        self.summary_config.reset()

        last_hiddens = torch.cat(last_hidden_state_list, dim=1)
        logits = self.lm_head(last_hiddens).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        past_key_values = outputs.past_key_values
        if past_key_values is not None:
            past_key_values = tuple(
                tuple(tensor[:, :, softprompt_length:tensor.size(-2)-summary_length] for tensor in tensors)
                for tensors in past_key_values
            )

        output = CausalACOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=output_hidden_states_list if output_hidden_states_list[0] is not None else None,
            attentions=output_attentions_list if output_attentions_list[0] is not None else None,
            softprompt=softprompt,
            softprompt_past_key_values=softprompt_past_key_values,
        )

        if return_dict:
            return output
        else:
            return tuple(output.values())
