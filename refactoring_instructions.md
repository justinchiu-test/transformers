# Model Refactoring Instructions for Shared Library

## Overview
This document provides instructions for refactoring 10 model implementations to use a shared library architecture.

**Important Note:** This refactoring will be a human-directed process. The implementation will be done interactively through chat, where you will provide specific instructions for each change, and I will implement them according to your guidance. This document serves as a roadmap and reference for the refactoring process.

## Target Models and File Paths

### 1. Qwen2
**Modeling Files:**
- `src/transformers/models/qwen2/modeling_qwen2.py`

**Test Files:**
- `tests/models/qwen2/test_modeling_qwen2.py`
- `tests/models/qwen2/test_tokenization_qwen2.py`

### 2. Llama
**Modeling Files:**
- `src/transformers/models/llama/modeling_llama.py`

**Test Files:**
- `tests/models/llama/test_modeling_llama.py`
- `tests/models/llama/test_tokenization_llama.py`

### 3. Gemma
**Modeling Files:**
- `src/transformers/models/gemma/modeling_gemma.py`

**Test Files:**
- `tests/models/gemma/test_modeling_gemma.py`
- `tests/models/gemma/test_tokenization_gemma.py`

### 4. Mistral
**Modeling Files:**
- `src/transformers/models/mistral/modeling_mistral.py`

**Test Files:**
- `tests/models/mistral/test_modeling_mistral.py`

### 5. Qwen2_VL
**Modeling Files:**
- `src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`

**Test Files:**
- `tests/models/qwen2_vl/test_modeling_qwen2_vl.py`
- `tests/models/qwen2_vl/test_image_processing_qwen2_vl.py`
- `tests/models/qwen2_vl/test_processing_qwen2_vl.py`
- `tests/models/qwen2_vl/test_video_processing_qwen2_vl.py`

### 6. GPT OSS
**Modeling Files:**
- `src/transformers/models/gpt_oss/modeling_gpt_oss.py`

**Test Files:**
- `tests/models/gpt_oss/test_modeling_gpt_oss.py`

### 7. DeepSeek V3
**Modeling Files:**
- `src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`

**Test Files:**
- `tests/models/deepseek_v3/test_modeling_deepseek_v3.py`

### 8. Mixtral
**Modeling Files:**
- `src/transformers/models/mixtral/modeling_mixtral.py`

**Test Files:**
- `tests/models/mixtral/test_modeling_mixtral.py`

### 9. OLMo2
**Modeling Files:**
- `src/transformers/models/olmo2/modeling_olmo2.py`

**Test Files:**
- `tests/models/olmo2/test_modeling_olmo2.py`

### 10. Qwen3
**Modeling Files:**
- `src/transformers/models/qwen3/modeling_qwen3.py`

**Test Files:**
- `tests/models/qwen3/test_modeling_qwen3.py`

## Refactoring Steps

**Note:** Each phase below will be executed based on your specific instructions in the chat. You will direct which components to refactor, when to proceed to the next phase, and any modifications to the approach.

### Phase 1: Analysis
1. **Identify Common Components**
   - Analyze each modeling file to identify shared components:
     - Attention mechanisms (RoPE, Flash Attention, etc.)
     - MLP/FFN layers
     - Layer normalization (RMSNorm, LayerNorm)
     - Positional embeddings
     - Cache utilities

2. **Document Model-Specific Features**
   - For each model, document unique features that need to be preserved:
     - Custom activation functions
     - Special attention patterns
     - Model-specific configurations

### Phase 2: Create Shared Library
1. **Create shared library structure:**
   ```
   src/transformers/models/shared/
   ├── __init__.py
   ├── attention.py          # Shared attention implementations
   ├── mlp.py                # Shared MLP/FFN implementations
   ├── normalization.py      # Shared normalization layers
   ├── embeddings.py         # Shared embedding layers
   ├── cache.py             # Shared cache utilities
   └── utils.py             # Common utility functions
   ```

2. **Extract common components:**
   - Start with the most commonly used components across all models
   - Create abstract base classes where appropriate
   - Implement configurable components that can handle model-specific variations

### Phase 3: Refactor Individual Models
**You will specify which model to refactor first and provide detailed instructions for each step.**

For each model, follow these steps:

1. **Import shared components:**
   ```python
   from ..shared import SharedAttention, SharedMLP, SharedRMSNorm
   from ..shared.embeddings import apply_rotary_pos_emb, rotate_half
   ```

2. **COMPLETELY REMOVE and replace duplicated code:**
   - **DELETE** the original attention class (e.g., `LlamaAttention`) and replace ALL references with `SharedAttention`
   - **DELETE** the original MLP class (e.g., `LlamaMLP`) and replace ALL references with `SharedMLP`
   - **DELETE** the original RMSNorm class (e.g., `LlamaRMSNorm`) and replace ALL references with `SharedRMSNorm`
   - **DELETE** the original RoPE functions (`rotate_half`, `apply_rotary_pos_emb`) and import from shared
   - **UPDATE** all class instantiations throughout the file (in DecoderLayer, Model classes, etc.)
   - **ENSURE** no duplicate implementations remain in the original file

3. **Update all references throughout the file:**
   ```python
   # Before:
   self.self_attn = LlamaAttention(config, layer_idx)
   self.mlp = LlamaMLP(config)
   self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

   # After:
   self.self_attn = SharedAttention(config, layer_idx)
   self.mlp = SharedMLP(config)
   self.input_layernorm = SharedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
   ```

4. **Maintain model-specific logic ONLY:**
   - Keep ONLY unique features that are NOT in shared components
   - Remove ALL duplicate implementations
   - The file should be significantly shorter after refactoring

5. **Verify complete refactoring:**
   - Search for old class names - they should NOT exist
   - Ensure ALL instances use shared components
   - Original model file should have ~200-300 fewer lines

### Phase 4: Testing
1. **Run existing tests (modeling files only, not tokenization) using uv:**
   ```bash
   # Using uv to run pytest
   uv run pytest tests/models/qwen2/test_modeling_qwen2.py
   uv run pytest tests/models/llama/test_modeling_llama.py
   uv run pytest tests/models/gemma/test_modeling_gemma.py
   uv run pytest tests/models/mistral/test_modeling_mistral.py
   uv run pytest tests/models/qwen2_vl/test_modeling_qwen2_vl.py
   uv run pytest tests/models/gpt_oss/test_modeling_gpt_oss.py
   uv run pytest tests/models/deepseek_v3/test_modeling_deepseek_v3.py
   uv run pytest tests/models/mixtral/test_modeling_mixtral.py
   uv run pytest tests/models/olmo2/test_modeling_olmo2.py
   uv run pytest tests/models/qwen3/test_modeling_qwen3.py

   # Or run all tests for a model at once:
   uv run python -m pytest -n auto --dist=loadfile -s -v ./tests/models/qwen2/test_modeling_qwen2.py
   ```

   **Note:** Tokenization tests are excluded from the testing scope.

2. **Verify backward compatibility:**
   - Load existing checkpoints
   - Compare outputs before and after refactoring
   - Ensure numerical precision is maintained

3. **Performance testing:**
   - Benchmark inference speed
   - Check memory usage
   - Verify training stability

### Phase 5: Documentation
1. **Update model documentation:**
   - Document which components are now shared
   - Update architecture diagrams if necessary

2. **Create migration guide:**
   - Document any breaking changes
   - Provide examples of how to use new shared components

## Key Considerations

### Backward Compatibility
- All refactored models must be able to load existing checkpoints
- Public API must remain unchanged
- Model outputs must be numerically identical (within tolerance)

### Performance
- Shared components should not introduce performance overhead
- Consider JIT compilation compatibility
- Maintain support for hardware-specific optimizations (CUDA, etc.)

### Maintainability
- Clear separation between shared and model-specific code
- Comprehensive docstrings for shared components
- Unit tests for all shared components

## Common Components to Extract

Based on initial analysis, these components appear across multiple models:

1. **RoPE (Rotary Position Embeddings)**
   - Used in: Qwen2, Llama, Mistral, DeepSeek V3, OLMo2, Qwen3

2. **RMSNorm**
   - Used in: Qwen2, Llama, Gemma, Mistral, DeepSeek V3, Mixtral, OLMo2, Qwen3

3. **SwiGLU/GatedMLP**
   - Used in: Qwen2, Llama, Mistral, DeepSeek V3, Mixtral, OLMo2, Qwen3

4. **Flash Attention Support**
   - Used in: Most models with configurable attention backends

5. **Cache Utilities**
   - Static/Dynamic cache implementations shared across models

## Testing Strategy

### Unit Tests for Shared Components
Create comprehensive tests for each shared component:
```
tests/models/shared/
├── test_attention.py
├── test_mlp.py
├── test_normalization.py
├── test_embeddings.py
└── test_cache.py
```

Run shared component tests using uv:
```bash
uv run pytest tests/models/shared/
```

### Integration Tests
- Ensure each refactored model passes its existing test suite
- Add regression tests comparing outputs before and after refactoring

### Performance Benchmarks
- Create benchmarks comparing performance before and after refactoring
- Monitor for any performance regressions

## Implementation Process

Since this is a human-directed refactoring:
1. **You will provide specific instructions** for which components to analyze first
2. **You will decide** when to move from analysis to implementation
3. **You will specify** which models to refactor and in what order
4. **You will guide** the testing and validation approach
5. **All changes will be made interactively** based on your directions in the chat

The timeline will depend on the pace and scope you set for each refactoring session.

## Success Criteria

- [ ] All 10 models refactored to use shared library
- [ ] **Original duplicate classes COMPLETELY REMOVED from model files**
- [ ] **Each model file reduced by ~200-300 lines**
- [ ] All existing tests pass
- [ ] No performance regression (< 5% tolerance)
- [ ] Backward compatibility maintained
- [ ] Code duplication reduced by at least 50% (~1400+ lines eliminated)
- [ ] **No duplicate implementations remain (verified by grep)**
- [ ] Documentation updated
- [ ] Peer review completed

## Verification Checklist

After refactoring each model, verify:
```bash
# These commands should return NO results after refactoring:
grep "class.*RMSNorm" src/transformers/models/{model_name}/modeling_{model_name}.py
grep "class.*MLP" src/transformers/models/{model_name}/modeling_{model_name}.py
grep "class.*Attention" src/transformers/models/{model_name}/modeling_{model_name}.py  # except for special variants
grep "def rotate_half" src/transformers/models/{model_name}/modeling_{model_name}.py
grep "def apply_rotary_pos_emb" src/transformers/models/{model_name}/modeling_{model_name}.py

# Line count should be reduced:
wc -l src/transformers/models/{model_name}/modeling_{model_name}.py  # Should be ~200-300 lines less
```
