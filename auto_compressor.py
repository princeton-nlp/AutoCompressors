import logging
import os
from typing import Optional, Union, List, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import OPTForCausalLM
from modeling_flash_llama import LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

import os

logger = logging.getLogger(__name__)

PastKVType = Optional[Tuple[Tuple[torch.FloatTensor]]]

@dataclass
class SummaryConfig:
    """Keep track of token constitution of current input sequence"""
    softprompt_length: int = 0
    past_key_values_softprompt_length: int = 0
    summary_length: int = 0

    def reset(self):
        self.softprompt_length = 0
        self.past_key_values_softprompt_length = 0
        self.summary_length = 0


@dataclass
class CausalACOutputWithPast(CausalLMOutputWithPast):
    softprompt: Optional[torch.FloatTensor]= None


class AutoCompressorMixin:
    """Mixin class to turn a AutoModelForCausalLM into an AutoCompressor."""

    def setup_autocompressor(self, config):
        """Call this function in the subclass __init__ to initialize the autocompressor. Override for custom behaviour"""
        assert hasattr(self.config, 'summary_length'), "Compressor requires a summary_length config parameter"

        self.summary_config = SummaryConfig()

        if config.summary_length > 0:
            self.embed_summary = nn.Embedding(config.summary_length, self.get_input_embeddings().embedding_dim)

            input_embeds = self.get_input_embeddings()
            self.embed_summary.weight.data[:,:] = (
                input_embeds.weight[config.eos_token_id]
            )

    def forward_segment(
        self,
        softprompt: torch.FloatTensor,
        segment_embeds: torch.FloatTensor,
        summary_token_embeds: torch.FloatTensor,
        segment_attention_mask: torch.LongTensor,
        past_key_values: PastKVType,
        output_hidden_states: bool,
        use_cache: bool,
        output_attentions: bool,
        segment_gradient_checkpointing: bool,
        past_key_values_softprompt_length: int
    ):

        bsz = segment_embeds.size(0)
        summary_length = summary_token_embeds.size(1)
        if past_key_values_softprompt_length > 0: # Softprompt should already be in past_key_values
            softprompt_length = 0
            segment_embeds = torch.cat([segment_embeds, summary_token_embeds], dim=1)

            device, attn_dtype = segment_embeds.device, segment_attention_mask.dtype
            segment_attention_mask = torch.cat([
                torch.ones(bsz, past_key_values_softprompt_length, device=device, dtype=attn_dtype),
                segment_attention_mask,
                torch.ones(bsz, summary_length, device=device, dtype=attn_dtype)
            ], dim=1)
        else:
            softprompt_length = softprompt.size(1)
            segment_embeds = torch.cat([softprompt, segment_embeds, summary_token_embeds], dim=1)

            device, attn_dtype = segment_embeds.device, segment_attention_mask.dtype
            segment_attention_mask = torch.cat([
                torch.ones(bsz, softprompt_length, device=device, dtype=attn_dtype),
                segment_attention_mask,
                torch.ones(bsz, summary_length, device=device, dtype=attn_dtype)
            ], dim=1)
        
        def decoder(segment_embeds,
                    segment_attention_mask,
                    segment_past_key_values,
                    softprompt_length,
                    past_key_values_softprompt_length,
                    summary_length):
            self.summary_config.softprompt_length = softprompt_length
            self.summary_config.past_key_values_softprompt_length = past_key_values_softprompt_length
            self.summary_config.summary_length = summary_length

            return self.model(
                inputs_embeds=segment_embeds,
                attention_mask=segment_attention_mask,
                past_key_values=segment_past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,)

        if segment_gradient_checkpointing:
            outputs = torch.utils.checkpoint.checkpoint(
                decoder, segment_embeds, segment_attention_mask, past_key_values,
                softprompt_length, past_key_values_softprompt_length, summary_length,
                use_reentrant=False)
        else:
            outputs = decoder(
                segment_embeds, segment_attention_mask, past_key_values,
                softprompt_length, past_key_values_softprompt_length, summary_length)

        total_length = outputs.last_hidden_state.size(1)
        segment_last_hiddens = (
            outputs.last_hidden_state[:, softprompt_length:total_length - summary_length]
        )
        new_softprompt = outputs.last_hidden_state[:, total_length - summary_length:]

        return outputs, segment_last_hiddens, new_softprompt

    def get_past_key_values_len(self, past_key_values):
        return 0 if past_key_values is None else past_key_values[0][0].size(2)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Union[PastKVType, Dict] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        segment_lengths: Optional[Union[List[int], int]] = None,
        softprompt: Optional[torch.FloatTensor] = None,
        output_softprompt: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # We formulate the past_key_values as a tuple where the second entry is the softprompt already in the past key values
        if past_key_values is not None and isinstance(past_key_values, dict):
            # Replace softprompt in direct argument with the softprompt in past_key_values
            past_key_values, softprompt = past_key_values["past_key_values"], past_key_values["softprompt"]
            past_key_values_softprompt_length = softprompt.size(1)
        else:
            past_key_values_softprompt_length = 0

        past_key_values_length = self.get_past_key_values_len(past_key_values) - past_key_values_softprompt_length

        if head_mask is not None:
            raise ValueError("Compressor does not support head_mask")
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Compressor does not support both input_ids and input_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if self.config.summary_length > 0:
            summary_token_ids = torch.arange(self.config.summary_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.size(0), -1)
            summary_token_embeds = self.embed_summary(summary_token_ids).to(inputs_embeds.dtype)
        else:
            summary_token_embeds = inputs_embeds[:,:0]

        # If no past_key_values are given, we will process the sequence in multiple segments
        if past_key_values is None:
            segment_lengths = segment_lengths if segment_lengths is not None else input_ids.size(1)

            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.size(0), inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device
                )

            inputs_embeds_list = torch.split(inputs_embeds, segment_lengths, dim=1)
            attention_mask_list = torch.split(attention_mask, segment_lengths, dim=1)
            summary_token_embeds_list = (
                (summary_token_embeds,) * (len(inputs_embeds_list) - 1) +
                (summary_token_embeds if output_softprompt else summary_token_embeds[:,:0,:],)
            )
        # With past_key_values we will process the input in a single pass (for generation), except when generting summary vectors
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.size(0), inputs_embeds.size(1) + past_key_values_length, dtype=torch.long, device=inputs_embeds.device
                )

            if use_cache and past_key_values_length + inputs_embeds.size(1) == segment_lengths:
                output_softprompt = True

                # If we use cache and output softprompt, we need to add a dummy segment to the end to get the past key values of the softprompt
                inputs_embeds_list = (inputs_embeds, inputs_embeds[:,:0,:])
                attention_mask_list = (attention_mask, attention_mask[:,:0])
                summary_token_embeds_list = (summary_token_embeds, summary_token_embeds[:,:0,:])
            else:
                inputs_embeds_list = (inputs_embeds,)
                attention_mask_list = (attention_mask,)
                summary_token_embeds_list = (summary_token_embeds if output_softprompt else summary_token_embeds[:,:0,:],)

        last_hidden_state_list = []
        output_attentions_list = []
        output_hidden_states_list = []

        if softprompt is None:
            softprompt = inputs_embeds[:,:0,:]

        for step, summary_token_embeds in enumerate(summary_token_embeds_list):
            is_last_step = step == len(inputs_embeds_list) - 1
            segment_gradient_checkpointing = (
                getattr(self.config, "segment_gradient_checkpointing", False) and
                self.training and not is_last_step
            )

            outputs, segment_hidden_states, new_softprompt = self.forward_segment(
                softprompt.to(inputs_embeds.dtype), inputs_embeds_list[step], summary_token_embeds, attention_mask_list[step],
                past_key_values, output_hidden_states, use_cache, output_attentions,
                segment_gradient_checkpointing, past_key_values_softprompt_length)

            last_hidden_state_list.append(segment_hidden_states)

            if self.config.accumulate_summary:
                softprompt = torch.cat([softprompt, new_softprompt], dim=1)
            elif new_softprompt.size(1) > 0:
                softprompt = new_softprompt

            output_attentions_list.append(outputs.attentions)
            output_hidden_states_list.append(outputs.hidden_states)

            # No past key values after first step
            past_key_values = None
            past_key_values_softprompt_length = 0

        # Output past values of last segment
        past_key_values = outputs.past_key_values

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

        output = CausalACOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values={"past_key_values": past_key_values, "softprompt": softprompt},
            hidden_states=output_hidden_states_list if output_hidden_states_list[0] is not None else None,
            attentions=output_attentions_list if output_attentions_list[0] is not None else None,
            softprompt=softprompt,
        )

        if return_dict:
            return output
        else:
            return tuple(output.values())

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        model_inputs["softprompt"] = kwargs.get("softprompt", None)
        model_inputs["segment_lengths"] = kwargs.get("segment_lengths", None)
        return model_inputs


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

        total_softprompt_length = self.summary_config.softprompt_length + self.summary_config.past_key_values_softprompt_length
        attention_mask = attention_mask[:, total_softprompt_length : attention_mask.size(1)-self.summary_config.summary_length]

        positions = attention_mask.cumsum(dim=1) * attention_mask + 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length - self.summary_config.past_key_values_softprompt_length :]
        positions = torch.cat([left_placeholder, positions, right_placeholder], dim=1)

        return super().forward(positions)


class OPTAutoCompressorModel(AutoCompressorMixin, OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.setup_autocompressor(config)

        # Custom positional embeddings
        self.model.decoder.embed_positions = OPTLearnedPositionalEmbeddingWithPadding(
            config.max_position_embeddings, config.hidden_size, summary_config=self.summary_config
        )

        # Initialize weights and apply final processing
        self.post_init()


# For backwards compatibility
AutoCompressorModel = OPTAutoCompressorModel


class LlamaAutoCompressorModel(AutoCompressorMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.setup_autocompressor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_past_key_values_len(self, past_key_values):
        # modeling_flash_llama has slightly different layout of past key vlaues
        return 0 if past_key_values is None else past_key_values[0][1]
