import torch
import torch.nn as nn


class SharedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, add_unit_offset=False):
        """
        SharedRMSNorm is equivalent to T5LayerNorm and used across multiple models

        Args:
            hidden_size: The size of the hidden states
            eps: The epsilon for numerical stability
            add_unit_offset: If True, apply (1.0 + weight) instead of weight (Gemma style)
        """
        super().__init__()
        self.add_unit_offset = add_unit_offset
        if add_unit_offset:
            # Gemma initializes weight to zeros when using (1.0 + weight)
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            # Standard initialization
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.add_unit_offset:
            # Gemma style: (1.0 + weight) * normalized_hidden_states
            # Also Gemma does (x * w).to(dtype) instead of x.to(dtype) * w
            output = hidden_states * (1.0 + self.weight.float())
            return output.to(input_dtype)
        else:
            # Standard style: weight * normalized_hidden_states
            return (self.weight.float() * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
