"""
Hierarchical memory system implementation with short-term and long-term storage.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

class MemoryEntry:
    def __init__(self, content: str, timestamp: float, embedding: Optional[np.ndarray] = None):
        self.content = content
        self.timestamp = timestamp
        self.embedding = embedding
        self.id = self._generate_id()
        
    def _generate_id(self) -> str:
        """Generate a unique ID for the memory entry."""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return f"{content_hash[:8]}_{int(self.timestamp)}"
    
    def to_dict(self) -> Dict:
        """Convert memory entry to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """Create memory entry from dictionary."""
        entry = cls(
            content=data["content"],
            timestamp=data["timestamp"],
            embedding=np.array(data["embedding"]) if data["embedding"] is not None else None
        )
        return entry

class ShortTermMemory:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.memories: List[MemoryEntry] = []
        
    def add(self, content: str, embedding: Optional[np.ndarray] = None) -> None:
        """Add a new memory to short-term storage."""
        entry = MemoryEntry(content, datetime.now().timestamp(), embedding)
        self.memories.append(entry)
        
        # Maintain max size
        if len(self.memories) > self.max_size:
            self.memories.pop(0)
    
    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """Get n most recent memories."""
        return self.memories[-n:]
    
    def clear(self) -> None:
        """Clear all short-term memories."""
        self.memories = []

class LongTermMemory:
    def __init__(self, storage_dir: str = "memory_store"):
        self.storage_dir = storage_dir
        self.ensure_storage_dir()
        self.memory_index: Dict[str, str] = {}
        self.load_index()
    
    def ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def load_index(self) -> None:
        """Load the memory index from disk."""
        index_path = os.path.join(self.storage_dir, "index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.memory_index = json.load(f)
    
    def save_index(self) -> None:
        """Save the memory index to disk."""
        index_path = os.path.join(self.storage_dir, "index.json")
        with open(index_path, 'w') as f:
            json.dump(self.memory_index, f)
    
    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry to long-term storage."""
        # Save memory entry
        file_path = os.path.join(self.storage_dir, f"{entry.id}.json")
        with open(file_path, 'w') as f:
            json.dump(entry.to_dict(), f)
        
        # Update index
        self.memory_index[entry.id] = file_path
        self.save_index()
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve most relevant memories based on embedding similarity."""
        if not self.memory_index:
            return []
        
        similarities = []
        for entry_id, file_path in self.memory_index.items():
            with open(file_path, 'r') as f:
                entry_data = json.load(f)
                entry = MemoryEntry.from_dict(entry_data)
                
                if entry.embedding is not None:
                    similarity = np.dot(query_embedding, entry.embedding)
                    similarities.append((similarity, entry))
        
        # Sort by similarity and return top-k
        similarities.sort(reverse=True)
        return [entry for _, entry in similarities[:top_k]]
    
    def delete(self, entry_id: str) -> None:
        """Delete a memory entry from storage."""
        if entry_id in self.memory_index:
            file_path = self.memory_index[entry_id]
            if os.path.exists(file_path):
                os.remove(file_path)
            del self.memory_index[entry_id]
            self.save_index()

class HierarchicalMemory:
    def __init__(self, short_term_size: int = 100, storage_dir: str = "memory_store"):
        self.short_term = ShortTermMemory(max_size=short_term_size)
        self.long_term = LongTermMemory(storage_dir=storage_dir)
    
    def add_memory(self, content: str, embedding: Optional[np.ndarray] = None) -> None:
        """Add a memory to both short-term and long-term storage."""
        # Add to short-term memory
        self.short_term.add(content, embedding)
        
        # Create and add to long-term memory
        entry = MemoryEntry(content, datetime.now().timestamp(), embedding)
        self.long_term.add(entry)
    
    def retrieve_relevant(self, query_embedding: np.ndarray, 
                         short_term_k: int = 3, long_term_k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories from both short-term and long-term storage."""
        # Get recent memories from short-term storage
        recent_memories = self.short_term.get_recent(short_term_k)
        
        # Get relevant memories from long-term storage
        relevant_memories = self.long_term.retrieve(query_embedding, top_k=long_term_k)
        
        # Combine and sort by timestamp
        all_memories = recent_memories + relevant_memories
        all_memories.sort(key=lambda x: x.timestamp, reverse=True)
        
        return all_memories
    
    def clear_short_term(self) -> None:
        """Clear short-term memory while preserving long-term storage."""
        self.short_term.clear() 