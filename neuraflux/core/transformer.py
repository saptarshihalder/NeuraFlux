"""
Core transformer implementation with custom attention and feed-forward layers.
"""

import numpy as np
from typing import Optional, Tuple, List
import math

class LayerNorm:
    def __init__(self, size: int, eps: float = 1e-5):
        self.size = size
        self.eps = eps
        self.gamma = np.ones(size)
        self.beta = np.zeros(size)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
    
    def backward(self, grad: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Simplified backward pass for demonstration
        return grad, np.zeros_like(self.gamma), np.zeros_like(self.beta)

class FeedForward:
    def __init__(self, dim: int, hidden_dim: int):
        self.w1 = np.random.normal(0, 0.02, (dim, hidden_dim))
        self.w2 = np.random.normal(0, 0.02, (hidden_dim, dim))
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(dim)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.hidden = np.maximum(0, np.dot(x, self.w1) + self.b1)
        return np.dot(self.hidden, self.w2) + self.b2
    
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Simplified backward pass for demonstration
        return np.zeros_like(self.w1), np.zeros_like(self.w2), np.zeros_like(self.b1), np.zeros_like(self.b2)

class TransformerBlock:
    def __init__(self, dim: int, num_heads: int, mlp_dim: int):
        self.attention = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim, mlp_dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Self-attention with residual connection
        attn_output = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn.forward(x)
        return self.norm2.forward(x + ffn_output)

class MultiHeadAttention:
    def __init__(self, dim: int, num_heads: int):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Initialize query, key, value projections
        self.q_proj = np.random.normal(0, 0.02, (dim, dim))
        self.k_proj = np.random.normal(0, 0.02, (dim, dim))
        self.v_proj = np.random.normal(0, 0.02, (dim, dim))
        self.out_proj = np.random.normal(0, 0.02, (dim, dim))
        
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size = q.shape[0]
        
        # Project queries, keys, values
        q = np.dot(q, self.q_proj)
        k = np.dot(k, self.k_proj)
        v = np.dot(v, self.v_proj)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
            
        attn = self.softmax(scores)
        out = np.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.num_heads * self.head_dim)
        return np.dot(out, self.out_proj)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class NeuralTransformer:
    def __init__(self, vocab_size: int, dim: int, num_heads: int, num_layers: int, 
                 mlp_dim: int, max_seq_len: int = 2048):
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Token embedding and positional encoding
        self.token_embedding = np.random.normal(0, 0.02, (vocab_size, dim))
        self.pos_embedding = self._create_positional_encoding(max_seq_len, dim)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(dim, num_heads, mlp_dim) for _ in range(num_layers)]
        
        # Output projection
        self.output_proj = np.random.normal(0, 0.02, (dim, vocab_size))
        
        # Layer normalization
        self.norm = LayerNorm(dim)
        
    def _create_positional_encoding(self, max_len: int, dim: int) -> np.ndarray:
        pos = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        pe = np.zeros((max_len, dim))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        return pe
    
    def forward(self, input_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Get embeddings
        x = self.token_embedding[input_ids]
        
        # Add positional encoding
        x = x + self.pos_embedding[:seq_len]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
            
        # Final layer norm and projection
        x = self.norm.forward(x)
        logits = np.dot(x, self.output_proj)
        
        return logits
    
    def generate(self, prompt_ids: np.ndarray, max_length: int = 100, 
                temperature: float = 0.7, top_k: int = 50) -> np.ndarray:
        """Generate text using the transformer model."""
        current_ids = prompt_ids.copy()
        
        for _ in range(max_length):
            # Get model predictions
            logits = self.forward(current_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            top_k_logits, top_k_indices = np.topk(next_token_logits, k=top_k, axis=-1)
            probs = self.softmax(top_k_logits)
            
            # Sample next token
            next_token_idx = np.random.choice(top_k, p=probs[0])
            next_token = top_k_indices[0, next_token_idx]
            
            # Append to sequence
            current_ids = np.append(current_ids, [[next_token]], axis=1)
            
        return current_ids
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True) 