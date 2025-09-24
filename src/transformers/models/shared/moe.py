"""Shared MoE (Mixture of Experts) implementation for transformer models."""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SharedRouter(nn.Module):
    """Shared router for MoE models supporting both softmax and sigmoid routing."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = getattr(config, 'num_local_experts', config.num_experts)
        self.num_experts_per_tok = getattr(config, 'num_experts_per_tok', config.top_k)
        self.hidden_dim = config.hidden_size

        # Router weights
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

        # Some models have router bias, others don't
        if getattr(config, 'router_bias', False):
            self.bias = nn.Parameter(torch.empty(self.num_experts))
        else:
            self.register_parameter('bias', None)

        # Routing style: 'softmax' (Mixtral/DeepSeek) or 'sigmoid' (GPT-OSS)
        self.routing_style = getattr(config, 'routing_style', 'softmax')

        # Handle GPT-OSS style detection
        if hasattr(config, 'model_type') and config.model_type == 'gpt_oss':
            self.routing_style = 'sigmoid'
            self.bias = nn.Parameter(torch.empty(self.num_experts))  # GPT-OSS always has bias

        # DeepSeek V3 specific configurations
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        self.topk_method = getattr(config, 'topk_method', 'greedy')
        self.n_group = getattr(config, 'n_group', None)
        self.topk_group = getattr(config, 'topk_group', None)
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', False)
        self.scoring_func = getattr(config, 'scoring_func', None)

        # Jitter noise for load balancing during training
        self.jitter_noise = getattr(config, 'jitter_noise', 0.0)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            routing_weights: Tensor of shape (batch*seq_len, num_experts_per_tok)
            selected_experts: Tensor of shape (batch*seq_len, num_experts_per_tok)
            router_logits: Raw router outputs for auxiliary loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Compute router logits
        router_logits = hidden_states @ self.weight.T
        if self.bias is not None:
            router_logits = router_logits + self.bias

        # Store raw logits for auxiliary loss computation
        raw_router_logits = router_logits

        # Apply jitter noise during training
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

        if self.routing_style == 'sigmoid':
            # GPT-OSS style: sigmoid routing
            router_probs = torch.sigmoid(router_logits)
        else:
            # Mixtral/DeepSeek style: softmax routing
            router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        if self.topk_method == 'group_limited_greedy' and self.n_group is not None:
            # DeepSeek V3 group-limited selection
            routing_weights, selected_experts = self._group_limited_topk(router_probs)
        else:
            # Standard top-k selection
            routing_weights, selected_experts = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)

        # Normalize routing weights if needed
        if self.norm_topk_prob and self.routing_style == 'softmax':
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Apply routed scaling factor (DeepSeek V3)
        routing_weights = routing_weights * self.routed_scaling_factor

        return routing_weights, selected_experts, raw_router_logits

    def _group_limited_topk(self, router_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """DeepSeek V3 group-limited top-k selection."""
        # This is a simplified version - full implementation would need more details
        # from DeepSeek V3 config
        group_size = self.num_experts // self.n_group
        group_scores = router_probs.view(-1, self.n_group, group_size).max(dim=-1).values

        # Select top groups
        _, top_groups = torch.topk(group_scores, self.topk_group, dim=-1)

        # Within selected groups, pick top experts
        # This is simplified - actual implementation would be more complex
        return torch.topk(router_probs, self.num_experts_per_tok, dim=-1)


class SharedMoE(nn.Module):
    """Shared MoE module combining router with expert networks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = getattr(config, 'num_local_experts', config.num_experts)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size

        # Router
        self.router = SharedRouter(config)

        # Expert networks
        self.experts = nn.ModuleList()

        # Determine expert type based on config
        expert_type = getattr(config, 'expert_type', 'standard')

        # Handle model-specific expert types
        if hasattr(config, 'model_type'):
            if config.model_type == 'gpt_oss':
                expert_type = 'gpt_oss'
            elif config.model_type == 'mixtral':
                expert_type = 'mixtral'
            elif config.model_type == 'deepseek_v3':
                expert_type = 'deepseek_v3'

        # Create experts based on type
        for _ in range(self.num_experts):
            if expert_type == 'gpt_oss':
                # GPT-OSS uses single fused gate_up and down projections
                from .mlp import SharedMLP
                self.experts.append(SharedMLP(config))
            else:
                # Mixtral/DeepSeek use standard MLP experts
                from .mlp import SharedMLP
                self.experts.append(SharedMLP(config))

        # Shared expert for DeepSeek V3
        self.has_shared_expert = getattr(config, 'use_shared_expert', False)
        if self.has_shared_expert:
            from .mlp import SharedMLP
            self.shared_expert = SharedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
            router_logits: Optional precomputed router logits

        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_dim)
            router_logits: Router logits for auxiliary loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get routing weights and selected experts
        routing_weights, selected_experts, router_logits = self.router(hidden_states)

        # Flatten input for expert processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Initialize output tensor
        output = torch.zeros_like(hidden_states_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = hidden_states_flat[expert_mask]

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get routing weights for this expert
                expert_weights = routing_weights[expert_mask]
                expert_weights = expert_weights[selected_experts[expert_mask] == expert_idx]

                # Weighted sum
                weighted_output = expert_output * expert_weights.unsqueeze(-1)

                # Add to output
                output[expert_mask] += weighted_output

        # Add shared expert output if present (DeepSeek V3)
        if self.has_shared_expert:
            shared_output = self.shared_expert(hidden_states_flat)
            output = output + shared_output

        # Reshape output
        output = output.view(batch_size, seq_len, hidden_dim)

        return output, router_logits