# Refactoring Analysis: Common Components Across 10 Models

## Analysis Summary

After analyzing all 10 models, here are the common components that can be extracted into a shared library:

## 1. Common Components Found

### 1.1 RMSNorm (Root Mean Square Normalization)
**Found in ALL models except Qwen2_VL (which has it embedded):**
- `Qwen2RMSNorm` (qwen2/modeling_qwen2.py:187)
- `LlamaRMSNorm` (llama/modeling_llama.py:53)
- `GemmaRMSNorm` (gemma/modeling_gemma.py:46)
- `MistralRMSNorm` (mistral/modeling_mistral.py:187)
- `GptOssRMSNorm` (gpt_oss/modeling_gpt_oss.py:47)
- `DeepseekV3RMSNorm` (deepseek_v3/modeling_deepseek_v3.py:36)
- `MixtralRMSNorm` (mixtral/modeling_mixtral.py:143)
- `Olmo2RMSNorm` (olmo2/modeling_olmo2.py:31)
- `Qwen3RMSNorm` (qwen3/modeling_qwen3.py:50)

**Pattern:** All implementations are nearly identical, computing `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`

### 1.2 Attention Mechanisms
**All models have attention classes:**
- `Qwen2Attention` (qwen2/modeling_qwen2.py:122)
- `LlamaAttention` (llama/modeling_llama.py:197)
- `GemmaAttention` (gemma/modeling_gemma.py:190)
- `MistralAttention` (mistral/modeling_mistral.py:123)
- `Qwen2VLAttention` (qwen2_vl/modeling_qwen2_vl.py:450)
- `GptOssAttention` (gpt_oss/modeling_gpt_oss.py:273)
- `DeepseekV3Attention` (deepseek_v3/modeling_deepseek_v3.py:327)
- `MixtralAttention` (mixtral/modeling_mixtral.py:235)
- `Olmo2Attention` (olmo2/modeling_olmo2.py:124)
- `Qwen3Attention` (qwen3/modeling_qwen3.py:158)

**Common features:**
- Multi-head attention mechanism
- RoPE (Rotary Position Embeddings) - found in ALL 10 models
- Flash Attention support
- Key-value caching

### 1.3 MLP/FFN Layers
**All models have MLP implementations:**
- `Qwen2MLP` (qwen2/modeling_qwen2.py:34)
- `LlamaMLP` (llama/modeling_llama.py:143)
- `GemmaMLP` (gemma/modeling_gemma.py:66)
- `MistralMLP` (mistral/modeling_mistral.py:35)
- `Qwen2MLP` (qwen2_vl/modeling_qwen2_vl.py:434)
- `GptOssMLP` (gpt_oss/modeling_gpt_oss.py:162)
- `DeepseekV3MLP` (deepseek_v3/modeling_deepseek_v3.py:92)
- `MixtralBlockSparseTop2MLP` (mixtral/modeling_mixtral.py:57) - **Special: MoE variant**
- `Olmo2MLP` (olmo2/modeling_olmo2.py:201)
- `Qwen3MLP` (qwen3/modeling_qwen3.py:70)

**Common pattern:** SwiGLU/GatedMLP with gate, up, and down projections

### 1.4 Rotary Position Embeddings (RoPE)
**Found in ALL 10 models:**
- Functions: `apply_rotary_pos_emb`, `rotate_half`
- Classes: Various RoPE/RotaryEmbedding implementations

## 2. Model-Specific Features

### 2.1 Mixtral
- **Unique:** Mixture of Experts (MoE) with `MixtralBlockSparseTop2MLP`
- Has router mechanism for expert selection

### 2.2 Qwen2_VL
- **Unique:** Vision components (`VisionAttention`, `VisionMlp`)
- Multimodal architecture with image processing

### 2.3 DeepSeek V3
- May have specialized MoE components (needs deeper analysis)

### 2.4 Gemma
- Uses `gelu_pytorch_tanh` activation (slightly different from others)

## 3. Proposed Shared Library Structure

```python
src/transformers/models/shared/
├── __init__.py
├── attention/
│   ├── __init__.py
│   ├── base.py              # BaseAttention class
│   ├── rope.py              # RoPE implementations
│   ├── flash_attention.py   # Flash attention utilities
│   └── cache.py             # KV cache utilities
├── mlp/
│   ├── __init__.py
│   ├── base.py              # BaseMLP class
│   ├── gated_mlp.py         # SwiGLU/Gated implementations
│   └── moe_mlp.py           # MoE MLP base (for Mixtral/DeepSeek)
├── normalization/
│   ├── __init__.py
│   └── rms_norm.py          # Shared RMSNorm
├── embeddings/
│   ├── __init__.py
│   └── rotary.py            # Shared RoPE implementations
└── utils/
    ├── __init__.py
    ├── activations.py        # Shared activation functions
    └── common.py             # Common utilities
```

## 4. Refactoring Strategy

### Phase 1: Start with RMSNorm (Simplest)
- Create `SharedRMSNorm` class
- Parameters: `hidden_size`, `eps=1e-6`
- Exact same implementation across all models

### Phase 2: Extract RoPE utilities
- Create shared `rotate_half` and `apply_rotary_pos_emb` functions
- Create configurable `RotaryEmbedding` class

### Phase 3: Base MLP class
- Create `BaseGatedMLP` with configurable:
  - Hidden size
  - Intermediate size
  - Activation function
  - Bias usage

### Phase 4: Base Attention class
- Create `BaseAttention` with:
  - Configurable heads
  - RoPE support
  - Flash attention support
  - KV cache handling

## 5. Implementation Order (Recommended)

1. **Start with Qwen2** - Clean, standard implementation
2. **Then Llama** - Very similar to Qwen2
3. **Then Mistral** - Also similar architecture
4. **Then Gemma** - Slight variations in activation
5. **Then GPT OSS** - Standard architecture
6. **Then OLMo2** - Standard with minor differences
7. **Then Qwen3** - Similar to Qwen2
8. **Then DeepSeek V3** - May have MoE components
9. **Then Mixtral** - MoE architecture needs special handling
10. **Finally Qwen2_VL** - Vision components need separate treatment

## 6. Testing Strategy

For each refactored component:
1. Create unit tests comparing outputs before/after refactoring
2. Test numerical precision (within 1e-5 tolerance)
3. Test backward compatibility with existing checkpoints
4. Performance benchmarks

## 7. Risk Assessment

**Low Risk:**
- RMSNorm extraction (identical across models)
- RoPE utilities (well-defined mathematical operations)

**Medium Risk:**
- MLP extraction (some variation in activations)
- Basic attention mechanisms

**High Risk:**
- MoE components (Mixtral, possibly DeepSeek V3)
- Vision components (Qwen2_VL)

## Next Steps

1. **Await your instructions** on which component to refactor first
2. **Create the shared library structure** as directed
3. **Implement shared components** one by one
4. **Refactor models incrementally** with your guidance
5. **Test each change** before proceeding

This analysis provides the foundation for the refactoring. Please provide specific instructions on which component you'd like to start with.