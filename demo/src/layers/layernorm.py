"""
RMSNorm implementation
参考 nano-vllm 实现
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard RMS normalization
        
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            Normalized tensor
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight
        return x
    
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        RMS normalization with residual addition
        
        Args:
            x: [batch, seq_len, hidden_size]
            residual: [batch, seq_len, hidden_size]
        
        Returns:
            Normalized tensor and new residual
        """
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight
        return x, residual
    
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            residual: Optional residual connection
        
        Returns:
            Normalized output (and new residual if residual provided)
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
