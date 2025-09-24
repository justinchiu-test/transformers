"""Shared decoder layer implementation for transformer models."""
from typing import Optional
import torch
from torch import nn
from ...modeling_layers import GradientCheckpointingLayer
from ...cache_utils import Cache
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg
from .attention import SharedAttention
from .mlp import SharedMLP
from .normalization import SharedRMSNorm


class SharedDecoderLayer(GradientCheckpointingLayer):
    """Shared decoder layer that supports both pre-norm and post-norm architectures."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention and MLP components
        self.self_attn = SharedAttention(config=config, layer_idx=layer_idx)
        self.mlp = SharedMLP(config)

        # Determine normalization style (pre-norm vs post-norm)
        self.use_post_norm = getattr(config, 'use_post_norm', False)
        # Special handling for OLMo2 which uses post-norm
        if hasattr(config, 'model_type') and config.model_type == 'olmo2':
            self.use_post_norm = True

        # RMSNorm configuration
        rms_eps = config.rms_norm_eps
        # Check for add_unit_offset in config, or detect Gemma model
        add_unit_offset = getattr(config, 'add_unit_offset', False)
        # Special handling for Gemma which uses add_unit_offset but doesn't have it in config
        if hasattr(config, 'model_type') and config.model_type == 'gemma':
            add_unit_offset = True

        if self.use_post_norm:
            # Post-norm style (OLMo2)
            self.post_attention_layernorm = SharedRMSNorm(
                config.hidden_size, eps=rms_eps, add_unit_offset=add_unit_offset
            )
            self.post_feedforward_layernorm = SharedRMSNorm(
                config.hidden_size, eps=rms_eps, add_unit_offset=add_unit_offset
            )
            self.input_layernorm = None
            self.post_mlp_layernorm = None
        else:
            # Pre-norm style (Llama, Mistral, Gemma, etc.)
            self.input_layernorm = SharedRMSNorm(
                config.hidden_size, eps=rms_eps, add_unit_offset=add_unit_offset
            )
            self.post_attention_layernorm = SharedRMSNorm(
                config.hidden_size, eps=rms_eps, add_unit_offset=add_unit_offset
            )
            self.post_mlp_layernorm = None
            self.post_feedforward_layernorm = None

        # Support for attention_type (used in Gemma, Qwen3, etc.)
        if hasattr(config, 'layer_types'):
            self.attention_type = config.layer_types[layer_idx]
        else:
            self.attention_type = None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:

        if self.use_post_norm:
            # Post-norm style (OLMo2)
            residual = hidden_states
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            mlp_output = self.mlp(hidden_states)
            # Handle MoE models that return (output, router_scores)
            if isinstance(mlp_output, tuple):
                hidden_states, _ = mlp_output
            else:
                hidden_states = mlp_output
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
        else:
            # Pre-norm style (Llama, Mistral, Gemma, etc.)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(hidden_states)
            # Handle MoE models that return (output, router_scores)
            if isinstance(mlp_output, tuple):
                hidden_states, _ = mlp_output
            else:
                hidden_states = mlp_output
            hidden_states = residual + hidden_states

        return hidden_states