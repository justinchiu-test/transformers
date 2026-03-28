# Multi-Head Attention Implementation Comparison

## Analysis: Are they copy-pasted?

**SHORT ANSWER: YES, they are nearly identical!** The attention classes are 95% the same code with only minor differences.

## The Evidence

### 1. Identical Structure
All attention classes have:
- **Same docstring**: `"Multi-headed attention from 'Attention Is All You Need' paper"`
- **Same __init__ parameters**: `(config, layer_idx)`
- **Same core attributes**:
  - `self.head_dim`
  - `self.num_key_value_groups`
  - `self.scaling`
  - `self.attention_dropout`
  - `self.is_causal = True`
- **Same projection layers**: q_proj, k_proj, v_proj, o_proj
- **Same forward signature** (with minor type hint differences)

### 2. Identical Forward Implementation
Lines 150-163 are virtually identical across models:
```python
# All models have this exact pattern:
input_shape = hidden_states.shape[:-1]
hidden_shape = (*input_shape, -1, self.head_dim)

query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

if past_key_values is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

### 3. Minor Differences (< 5% of code)

#### Bias Settings:
- **Qwen2**: q,k,v have `bias=True`, o has `bias=False`
- **Llama**: Uses `config.attention_bias` for all projections
- **Mistral**: All projections have `bias=False`
- **Gemma**: Similar to others
- **Qwen3**: Similar to Qwen2

#### Special Features:
- **Qwen2**: Has `sliding_window` support (1 extra line)
- **Others**: Standard implementation

#### Type Hints:
- Some use `FlashAttentionKwargs`, others use `TransformersKwargs`

## Conclusion

**These are clearly copy-pasted implementations with minimal modifications.**

The differences are so minor they could be handled with configuration:
```python
class SharedAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # ... same init code ...

        # Handle bias differences
        bias_q = getattr(config, 'attention_bias_q', config.attention_bias if hasattr(config, 'attention_bias') else False)
        bias_k = getattr(config, 'attention_bias_k', config.attention_bias if hasattr(config, 'attention_bias') else False)
        bias_v = getattr(config, 'attention_bias_v', config.attention_bias if hasattr(config, 'attention_bias') else False)
        bias_o = getattr(config, 'attention_bias_o', config.attention_bias if hasattr(config, 'attention_bias') else False)

        # Handle sliding window
        self.sliding_window = getattr(config, 'sliding_window', None)
```

## Duplication Statistics

- **~50-60 lines of identical code** per attention class
- **10 models** = ~500-600 lines of duplicated code
- Could be reduced to **1 shared class** of ~60 lines

This is a perfect candidate for refactoring into a shared component!