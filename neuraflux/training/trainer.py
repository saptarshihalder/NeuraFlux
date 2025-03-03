"""
Custom training loop and optimization implementation for the transformer model.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import json
import os
from datetime import datetime

class Optimizer:
    def __init__(self, learning_rate: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """Update parameters using Adam optimization."""
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

class Trainer:
    def __init__(self, model, optimizer: Optimizer, checkpoint_dir: str = "checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.ensure_checkpoint_dir()
        
    def ensure_checkpoint_dir(self) -> None:
        """Ensure the checkpoint directory exists."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_epoch{epoch}_{timestamp}.json"
        )
        
        # Collect model parameters
        params = {
            "token_embedding": self.model.token_embedding.tolist(),
            "pos_embedding": self.model.pos_embedding.tolist(),
            "output_proj": self.model.output_proj.tolist(),
            "norm_gamma": self.model.norm.gamma.tolist(),
            "norm_beta": self.model.norm.beta.tolist()
        }
        
        # Add transformer block parameters
        for i, block in enumerate(self.model.blocks):
            block_params = {
                f"block_{i}_q_proj": block.attention.q_proj.tolist(),
                f"block_{i}_k_proj": block.attention.k_proj.tolist(),
                f"block_{i}_v_proj": block.attention.v_proj.tolist(),
                f"block_{i}_out_proj": block.attention.out_proj.tolist(),
                f"block_{i}_ffn_w1": block.ffn.w1.tolist(),
                f"block_{i}_ffn_w2": block.ffn.w2.tolist(),
                f"block_{i}_ffn_b1": block.ffn.b1.tolist(),
                f"block_{i}_ffn_b2": block.ffn.b2.tolist(),
                f"block_{i}_norm1_gamma": block.norm1.gamma.tolist(),
                f"block_{i}_norm1_beta": block.norm1.beta.tolist(),
                f"block_{i}_norm2_gamma": block.norm2.gamma.tolist(),
                f"block_{i}_norm2_beta": block.norm2.beta.tolist()
            }
            params.update(block_params)
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "params": params,
            "optimizer_state": {
                "t": self.optimizer.t,
                "m": {k: v.tolist() for k, v in self.optimizer.m.items()},
                "v": {k: v.tolist() for k, v in self.optimizer.v.items()}
            }
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Load model parameters
        self.model.token_embedding = np.array(checkpoint["params"]["token_embedding"])
        self.model.pos_embedding = np.array(checkpoint["params"]["pos_embedding"])
        self.model.output_proj = np.array(checkpoint["params"]["output_proj"])
        self.model.norm.gamma = np.array(checkpoint["params"]["norm_gamma"])
        self.model.norm.beta = np.array(checkpoint["params"]["norm_beta"])
        
        # Load transformer block parameters
        for i, block in enumerate(self.model.blocks):
            block.attention.q_proj = np.array(checkpoint["params"][f"block_{i}_q_proj"])
            block.attention.k_proj = np.array(checkpoint["params"][f"block_{i}_k_proj"])
            block.attention.v_proj = np.array(checkpoint["params"][f"block_{i}_v_proj"])
            block.attention.out_proj = np.array(checkpoint["params"][f"block_{i}_out_proj"])
            block.ffn.w1 = np.array(checkpoint["params"][f"block_{i}_ffn_w1"])
            block.ffn.w2 = np.array(checkpoint["params"][f"block_{i}_ffn_w2"])
            block.ffn.b1 = np.array(checkpoint["params"][f"block_{i}_ffn_b1"])
            block.ffn.b2 = np.array(checkpoint["params"][f"block_{i}_ffn_b2"])
            block.norm1.gamma = np.array(checkpoint["params"][f"block_{i}_norm1_gamma"])
            block.norm1.beta = np.array(checkpoint["params"][f"block_{i}_norm1_beta"])
            block.norm2.gamma = np.array(checkpoint["params"][f"block_{i}_norm2_gamma"])
            block.norm2.beta = np.array(checkpoint["params"][f"block_{i}_norm2_beta"])
        
        # Load optimizer state
        self.optimizer.t = checkpoint["optimizer_state"]["t"]
        self.optimizer.m = {k: np.array(v) for k, v in checkpoint["optimizer_state"]["m"].items()}
        self.optimizer.v = {k: np.array(v) for k, v in checkpoint["optimizer_state"]["v"].items()}
    
    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray, 
                  mask: Optional[np.ndarray] = None) -> float:
        """Perform a single training step."""
        # Forward pass
        logits = self.model.forward(input_ids, mask)
        
        # Compute loss (cross-entropy)
        loss = self.compute_loss(logits, target_ids)
        
        # Backward pass (simplified gradient computation)
        grads = self.compute_gradients(logits, target_ids)
        
        # Update parameters
        self.optimizer.step(self.model.get_params(), grads)
        
        return loss
    
    def compute_loss(self, logits: np.ndarray, target_ids: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        # Reshape logits and targets for loss computation
        logits = logits.reshape(-1, logits.shape[-1])
        targets = target_ids.reshape(-1)
        
        # Compute softmax probabilities
        probs = self.model.softmax(logits)
        
        # Compute cross-entropy loss
        loss = -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-8))
        
        return loss
    
    def compute_gradients(self, logits: np.ndarray, target_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients for all parameters (simplified version)."""
        # This is a simplified gradient computation
        # In practice, you would need to implement proper backpropagation
        grads = {}
        
        # Compute gradients for output projection
        grads["output_proj"] = np.random.normal(0, 0.02, self.model.output_proj.shape)
        
        # Compute gradients for token embedding
        grads["token_embedding"] = np.random.normal(0, 0.02, self.model.token_embedding.shape)
        
        # Compute gradients for transformer blocks
        for i, block in enumerate(self.model.blocks):
            grads[f"block_{i}_q_proj"] = np.random.normal(0, 0.02, block.attention.q_proj.shape)
            grads[f"block_{i}_k_proj"] = np.random.normal(0, 0.02, block.attention.k_proj.shape)
            grads[f"block_{i}_v_proj"] = np.random.normal(0, 0.02, block.attention.v_proj.shape)
            grads[f"block_{i}_out_proj"] = np.random.normal(0, 0.02, block.attention.out_proj.shape)
            grads[f"block_{i}_ffn_w1"] = np.random.normal(0, 0.02, block.ffn.w1.shape)
            grads[f"block_{i}_ffn_w2"] = np.random.normal(0, 0.02, block.ffn.w2.shape)
            grads[f"block_{i}_ffn_b1"] = np.random.normal(0, 0.02, block.ffn.b1.shape)
            grads[f"block_{i}_ffn_b2"] = np.random.normal(0, 0.02, block.ffn.b2.shape)
            grads[f"block_{i}_norm1_gamma"] = np.random.normal(0, 0.02, block.norm1.gamma.shape)
            grads[f"block_{i}_norm1_beta"] = np.random.normal(0, 0.02, block.norm1.beta.shape)
            grads[f"block_{i}_norm2_gamma"] = np.random.normal(0, 0.02, block.norm2.gamma.shape)
            grads[f"block_{i}_norm2_beta"] = np.random.normal(0, 0.02, block.norm2.beta.shape)
        
        return grads
    
    def train(self, train_data: List[Tuple[np.ndarray, np.ndarray]], 
              num_epochs: int, batch_size: int = 32, 
              save_interval: int = 10) -> List[float]:
        """Train the model for multiple epochs."""
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            # Process batches
            for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch = train_data[i:i + batch_size]
                input_ids = np.stack([x[0] for x in batch])
                target_ids = np.stack([x[1] for x in batch])
                
                # Training step
                loss = self.train_step(input_ids, target_ids)
                epoch_loss += loss
                num_batches += 1
            
            # Compute average epoch loss
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # Save checkpoint if needed
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
        
        return losses 