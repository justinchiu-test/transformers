"""Shared base model classes for transformer models."""
from typing import Optional, Union, Callable
import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from .decoder_layer import SharedDecoderLayer
from .normalization import SharedRMSNorm
from .embeddings import SharedRotaryEmbedding


class SharedPreTrainedModel(PreTrainedModel):
    """Shared base class for PreTrainedModel with common settings."""
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        # Set model-specific attributes from config
        if hasattr(config, 'model_type'):
            model_type = config.model_type
            # Set _no_split_modules based on model type
            decoder_layer_map = {
                'llama': 'LlamaDecoderLayer',
                'mistral': 'MistralDecoderLayer',
                'gemma': 'GemmaDecoderLayer',
                'gemma2': 'Gemma2DecoderLayer',
                'qwen2': 'Qwen2DecoderLayer',
                'qwen2_vl': 'Qwen2VLDecoderLayer',
                'olmo2': 'Olmo2DecoderLayer',
                'gpt_oss': 'GptOssDecoderLayer',
                'mixtral': 'MixtralDecoderLayer',
                'deepseek_v3': 'DeepseekV3DecoderLayer',
            }
            if model_type in decoder_layer_map:
                self._no_split_modules = [decoder_layer_map[model_type]]

            # Model-specific flags
            if model_type == 'mixtral':
                self._can_compile_fullgraph = False  # MoE models don't work with torch.compile
            elif model_type == 'deepseek_v3':
                self._can_compile_fullgraph = False
            else:
                self._can_compile_fullgraph = True


class SharedModel(SharedPreTrainedModel):
    """Shared base model class for decoder-only transformer models."""

    def __init__(self, config, decoder_layer_class=None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Decoder layers - use provided class or default to SharedDecoderLayer
        if decoder_layer_class is None:
            decoder_layer_class = SharedDecoderLayer

        self.layers = nn.ModuleList(
            [decoder_layer_class(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final normalization
        rms_eps = config.rms_norm_eps
        add_unit_offset = getattr(config, 'add_unit_offset', False)
        # Special handling for Gemma
        if hasattr(config, 'model_type') and config.model_type == 'gemma':
            add_unit_offset = True
        self.norm = SharedRMSNorm(config.hidden_size, eps=rms_eps, add_unit_offset=add_unit_offset)

        # Rotary embeddings
        self.rotary_emb = SharedRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Initialize lists to store outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            # Check if layer returns tuple (for attention weights)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
            else:
                hidden_states = layer_outputs
                if output_attentions:
                    # Layer doesn't return attentions, add None
                    all_self_attns += (None,)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SharedForCausalLM:
    """Shared base class for causal language modeling.

    This is a mixin class that should be inherited along with a PreTrainedModel
    and GenerationMixin by the actual model class.
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )