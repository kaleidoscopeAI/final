import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention with optional masking and dropout.
    
    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, dim]
        mask: Optional attention mask
        dropout: Optional dropout layer
    
    Returns:
        Tuple of (output tensor, attention weights)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights