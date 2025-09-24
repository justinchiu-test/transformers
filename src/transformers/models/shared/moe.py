"""Shared MoE (Mixture of Experts) implementation for transformer models."""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SharedRouter(nn.Module):
    """Shared router for MoE models based on GPT-OSS's clean implementation."""

    def __init__(self, config):
        super().__init__()
        self.top_k = getattr(config, 'num_experts_per_tok', getattr(config, 'top_k', 2))
        self.num_experts = getattr(config, 'num_local_experts', getattr(config, 'num_experts', 8))
        self.hidden_dim = config.hidden_size

        # Router weights (all models have this)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

        # Router bias (GPT-OSS has it, others can configure)
        if getattr(config, 'router_bias', True) if hasattr(config, 'model_type') and config.model_type == 'gpt_oss' else False:
            self.bias = nn.Parameter(torch.empty(self.num_experts))
        else:
            self.register_parameter('bias', None)

        # Mixtral-specific: jitter noise
        self.jitter_noise = getattr(config, 'router_jitter_noise', 0.0)

        # DeepSeek V3 specific configurations
        self.use_sigmoid = getattr(config, 'use_sigmoid_routing', False)
        if hasattr(config, 'model_type') and config.model_type == 'deepseek_v3':
            self.use_sigmoid = True
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', False)

        # DeepSeek V3 group selection (advanced feature)
        self.n_group = getattr(config, 'n_group', None)
        self.topk_group = getattr(config, 'topk_group', None)
        if self.n_group:
            self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPT-OSS style forward pass - clean and efficient.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)

        Returns:
            router_scores: Tensor of shape (seq_len, num_experts) with routing weights
            router_indices: Tensor of shape (seq_len, top_k) with selected expert indices
        """
        # Reshape to 2D if needed
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        # Mixtral: Apply jitter noise during training
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )

        # Compute router logits (GPT-OSS style with F.linear)
        router_logits = F.linear(hidden_states, self.weight, self.bias)

        # DeepSeek V3: Group-limited selection
        if self.n_group and self.topk_group:
            router_indices = self._get_topk_indices_grouped(router_logits)
            if self.use_sigmoid:
                scores = torch.sigmoid(router_logits)
            else:
                scores = router_logits
            router_top_values = scores.gather(1, router_indices)
        else:
            # Standard top-k selection (GPT-OSS style)
            router_top_values, router_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # Apply activation and normalization
        if self.use_sigmoid:
            # DeepSeek V3: Use sigmoid on values
            router_top_values = torch.sigmoid(router_top_values) if not self.n_group else router_top_values
        else:
            # GPT-OSS/Mixtral: Softmax over selected top-k values
            router_top_values = F.softmax(router_top_values, dim=1, dtype=router_top_values.dtype)

        # DeepSeek V3 & Mixtral: Optional normalization
        if self.norm_topk_prob or (hasattr(original_shape, '__len__') and len(original_shape) == 3 and
                                   hasattr(self, 'model_type') and self.model_type == 'mixtral'):
            denominator = router_top_values.sum(dim=-1, keepdim=True) + 1e-20
            router_top_values = router_top_values / denominator

        # Apply scaling factor if configured (DeepSeek V3)
        router_top_values = router_top_values * self.routed_scaling_factor

        # Create sparse routing matrix (GPT-OSS style)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_values)

        return router_scores, router_indices

    @torch.no_grad()
    def _get_topk_indices_grouped(self, scores: torch.Tensor) -> torch.Tensor:
        """DeepSeek V3 group-limited top-k selection."""
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Group scoring
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.num_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]

        # Create group mask
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.num_experts // self.n_group)
            .reshape(-1, self.num_experts)
        )

        # Mask and select top-k
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float('-inf'))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices


class SharedExperts(nn.Module):
    """
    GPT-OSS style expert module - efficient and clean.
    Can be configured for Mixtral and DeepSeek V3 compatibility.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = getattr(config, 'num_local_experts', getattr(config, 'num_experts', 8))
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, 'moe_intermediate_size', config.intermediate_size)

        # Determine expert architecture
        self.use_fused_experts = getattr(config, 'use_fused_experts', False)
        if hasattr(config, 'model_type') and config.model_type == 'gpt_oss':
            self.use_fused_experts = True

        if self.use_fused_experts:
            # GPT-OSS style: fused gate_up projection
            self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
            self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size))
            self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))
            self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

            # GPT-OSS specific activation parameters
            self.alpha = getattr(config, 'moe_alpha', 1.702)
            self.limit = getattr(config, 'moe_limit', 7.0)
        else:
            # Mixtral/DeepSeek style: separate expert MLPs
            from .mlp import SharedMLP
            self.experts = nn.ModuleList([
                SharedMLP(config) for _ in range(self.num_experts)
            ])

    def forward(self, hidden_states: torch.Tensor, router_indices: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        GPT-OSS style expert processing - clean and efficient.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size) or (seq_len, hidden_size)
            router_indices: (seq_len, top_k) - which experts to use
            routing_weights: (seq_len, num_experts) - sparse routing weights

        Returns:
            Output tensor of same shape as hidden_states
        """
        # Handle different input shapes
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_size)
            reshape_output = True
        else:
            batch_size = 1
            seq_len = hidden_states.shape[0]
            hidden_size = hidden_states.shape[1]
            reshape_output = False

        num_tokens = hidden_states.shape[0]
        num_experts = routing_weights.shape[1]

        if self.use_fused_experts:
            # GPT-OSS implementation
            next_states = torch.zeros_like(hidden_states)

            # Create expert mask for efficient processing
            expert_mask = F.one_hot(router_indices, num_classes=num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)

            # Find which experts are active
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == num_experts:  # skip padding index
                    continue

                # Find tokens for this expert
                _, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]

                # Fused gate_up projection
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]

                # GPT-OSS activation with clamping
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu

                # Down projection
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]

                # Apply routing weights and accumulate
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        else:
            # Mixtral/DeepSeek style
            next_states = torch.zeros_like(hidden_states)

            # One-hot encode selected experts
            expert_mask = F.one_hot(router_indices, num_classes=num_experts).permute(2, 1, 0)

            for expert_idx in range(num_experts):
                # Check if this expert is used
                expert_mask_idx = expert_mask[expert_idx]
                if not expert_mask_idx.any():
                    continue

                idx, top_x = torch.where(expert_mask_idx)

                # Get tokens for this expert
                current_state = hidden_states[top_x]

                # Apply expert MLP
                expert_output = self.experts[expert_idx](current_state)

                # Apply routing weights
                weighted_output = expert_output * routing_weights[top_x, expert_idx, None]

                # Accumulate output
                next_states.index_add_(0, top_x, weighted_output.to(hidden_states.dtype))

        # Reshape if needed
        if reshape_output:
            next_states = next_states.view(batch_size, seq_len, hidden_size)

        return next_states


class SharedMoE(nn.Module):
    """
    Complete MoE module based on GPT-OSS design.
    Combines router and experts with optional shared experts (DeepSeek V3).
    """

    def __init__(self, config):
        super().__init__()
        self.router = SharedRouter(config)
        self.experts = SharedExperts(config)

        # DeepSeek V3: Optional shared experts
        self.use_shared_experts = getattr(config, 'use_shared_experts', False)
        if self.use_shared_experts:
            from .mlp import SharedMLP
            n_shared = getattr(config, 'n_shared_experts', 2)
            shared_intermediate_size = config.intermediate_size * n_shared
            # Create a config for shared expert with larger intermediate size
            shared_config = type(config)(**{**config.__dict__, 'intermediate_size': shared_intermediate_size})
            self.shared_experts = SharedMLP(shared_config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass following GPT-OSS's clean design.

        Args:
            hidden_states: Input tensor

        Returns:
            output: Processed tensor
            router_scores: Router scores for auxiliary loss
        """
        # Route tokens to experts
        router_scores, router_indices = self.router(hidden_states)

        # Process through routed experts
        routed_output = self.experts(hidden_states, router_indices, router_scores)

        # Add shared experts if configured (DeepSeek V3)
        if self.use_shared_experts:
            shared_output = self.shared_experts(hidden_states)
            output = routed_output + shared_output
        else:
            output = routed_output

        return output, router_scores