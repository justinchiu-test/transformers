from typing import Callable, Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg
from .embeddings import apply_rotary_pos_emb


def eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    is_causal: bool = False,
    **kwargs,
):
    """Basic eager attention implementation - simplified version"""
    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SharedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Handle different attribute names across models
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Handle different bias configurations
        # Qwen2: q,k,v have bias=True, o has bias=False
        # Llama: uses config.attention_bias for all
        # Mistral: all have bias=False
        if hasattr(config, 'attention_bias'):
            # Llama-style
            q_bias = k_bias = v_bias = o_bias = config.attention_bias
        elif hasattr(config, 'qkv_bias'):
            # Potential Qwen2-style
            q_bias = k_bias = v_bias = config.qkv_bias
            o_bias = getattr(config, 'o_proj_bias', False)
        else:
            # Default (Mistral-style or detect from model name)
            model_type = getattr(config, 'model_type', '')
            if 'qwen2' in model_type.lower():
                q_bias = k_bias = v_bias = True
                o_bias = False
            else:
                q_bias = k_bias = v_bias = o_bias = False

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=q_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=k_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=v_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=o_bias)

        # Handle sliding window for Qwen2
        if hasattr(config, 'sliding_window') and hasattr(config, 'layer_types'):
            self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        else:
            self.sliding_window = getattr(config, 'sliding_window', None)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights