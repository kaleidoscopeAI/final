import numpy as np
import torch
from typing import Optional, Dict, Any
from functools import lru_cache

@lru_cache(maxsize=1024)
def compute_quantum_basis(n_qubits: int) -> np.ndarray:
    """
    Compute quantum basis states with caching for better performance.
    
    Args:
        n_qubits: Number of qubits in the system
    
    Returns:
        Precomputed basis states
    """
    return np.eye(2**n_qubits, dtype=np.float32)

class MemoryCache:
    """Memory-efficient caching system with size limits and LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[Any, Any] = {}
        self._access_count: Dict[Any, int] = {}
        self._current_count = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache with access counting."""
        if key in self._cache:
            self._access_count[key] = self._current_count
            self._current_count += 1
            return self._cache[key]
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache with size management."""
        if len(self._cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self._access_count.items(), key=lambda x: x[1])[0]
            self._cache.pop(lru_key)
            self._access_count.pop(lru_key)
            
        self._cache[key] = value
        self._access_count[key] = self._current_count
        self._current_count += 1
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_count.clear()
        self._current_count = 0

def optimize_memory_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Optimize memory layout for better performance."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor