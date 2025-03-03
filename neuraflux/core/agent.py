"""
Main agent class that integrates all components of the NeuraFlux system.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
import logging
from datetime import datetime

from .transformer import NeuralTransformer
from .memory import HierarchicalMemory
from ..tools.web_scraper import WebScraper
from ..tools.code_executor import CodeExecutor
from ..tools.math_engine import MathEngine
from ..training.trainer import Optimizer, Trainer

class AutonomousAgent:
    def __init__(self, model_path: Optional[str] = None):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model = self._initialize_model()
        self.memory = HierarchicalMemory()
        self.web_scraper = WebScraper()
        self.code_executor = CodeExecutor()
        self.math_engine = MathEngine()
        
        # Initialize optimizer and trainer
        self.optimizer = Optimizer()
        self.trainer = Trainer(self.model, self.optimizer)
        
        # Load model if path provided
        if model_path:
            self._load_model(model_path)
        
        # Initialize conversation history
        self.conversation_history: List[Dict] = []
        
        # Initialize learning parameters
        self.learning_rate = 1e-4
        self.max_context_length = 2048
        self.temperature = 0.7
        self.top_k = 50
        
        # Initialize persona
        self.persona = {
            'name': 'NeuraFlux',
            'traits': ['helpful', 'knowledgeable', 'friendly'],
            'expertise': ['programming', 'mathematics', 'general knowledge']
        }
    
    def _initialize_model(self) -> NeuralTransformer:
        """Initialize the transformer model with default parameters."""
        return NeuralTransformer(
            vocab_size=50000,
            dim=512,
            num_heads=8,
            num_layers=6,
            mlp_dim=2048
        )
    
    def _load_model(self, model_path: str) -> None:
        """Load a trained model from disk."""
        try:
            self.trainer.load_checkpoint(model_path)
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate a response."""
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Extract command if present
        if user_input.startswith('!'):
            response = self._handle_command(user_input)
        else:
            # Generate response using the model
            response = self._generate_response(user_input)
        
        # Add response to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Store in memory
        self.memory.add_memory(user_input)
        self.memory.add_memory(response)
        
        return response
    
    def _handle_command(self, command: str) -> str:
        """Handle special commands."""
        cmd = command.split()[0]
        args = command.split()[1:]
        
        if cmd == '!web_search':
            query = ' '.join(args)
            results = self.web_scraper.search(query)
            return self._format_search_results(results)
        
        elif cmd == '!execute_code':
            code = ' '.join(args)
            result = self.code_executor.execute(code)
            return self._format_code_result(result)
        
        elif cmd == '!solve_math':
            equation = ' '.join(args)
            try:
                solutions = self.math_engine.solve_equation(equation, 'x')
                return f"Solutions: {solutions}"
            except Exception as e:
                return f"Error solving equation: {str(e)}"
        
        elif cmd == '!memory_stats':
            return self._get_memory_stats()
        
        elif cmd == '!adjust_persona':
            return self._adjust_persona(args)
        
        else:
            return f"Unknown command: {cmd}"
    
    def _generate_response(self, user_input: str) -> str:
        """Generate a response using the transformer model."""
        # Convert input to token IDs
        input_ids = self._tokenize_input(user_input)
        
        # Generate response
        output_ids = self.model.generate(
            input_ids,
            max_length=100,
            temperature=self.temperature,
            top_k=self.top_k
        )
        
        # Convert back to text
        return self._detokenize_output(output_ids)
    
    def _tokenize_input(self, text: str) -> np.ndarray:
        """Convert text to token IDs (simplified)."""
        # This is a simplified tokenization
        # In practice, you would use a proper tokenizer
        return np.array([[hash(word) % self.model.vocab_size for word in text.split()]])
    
    def _detokenize_output(self, token_ids: np.ndarray) -> str:
        """Convert token IDs back to text (simplified)."""
        # This is a simplified detokenization
        # In practice, you would use a proper detokenizer
        return " ".join([str(id) for id in token_ids[0]])
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """Format web search results."""
        if not results:
            return "No results found."
        
        formatted = "Search Results:\n"
        for i, result in enumerate(results, 1):
            formatted += f"\n{i}. {result.get('title', 'No title')}\n"
            formatted += f"   {result.get('content', '')[:200]}...\n"
        
        return formatted
    
    def _format_code_result(self, result: Dict) -> str:
        """Format code execution results."""
        if result['success']:
            return f"Output:\n{result['output']}"
        else:
            return f"Error:\n{result['error']}"
    
    def _get_memory_stats(self) -> str:
        """Get memory statistics."""
        return f"""Memory Statistics:
Short-term memories: {len(self.memory.short_term.memories)}
Long-term memories: {len(self.memory.long_term.memory_index)}
Conversation history: {len(self.conversation_history)} messages"""
    
    def _adjust_persona(self, args: List[str]) -> str:
        """Adjust the agent's persona."""
        if not args:
            return "Please specify what aspect of the persona to adjust."
        
        aspect = args[0]
        value = ' '.join(args[1:])
        
        if aspect in self.persona:
            if isinstance(self.persona[aspect], list):
                self.persona[aspect] = value.split(',')
            else:
                self.persona[aspect] = value
            return f"Persona {aspect} updated successfully."
        else:
            return f"Unknown persona aspect: {aspect}"
    
    def learn_from_interaction(self, user_input: str, response: str, 
                             feedback: Optional[float] = None) -> None:
        """Learn from user interaction and feedback."""
        # Convert interaction to training data
        input_ids = self._tokenize_input(user_input)
        target_ids = self._tokenize_input(response)
        
        # Compute loss based on feedback if provided
        if feedback is not None:
            # Adjust model parameters based on feedback
            self._update_from_feedback(feedback)
        
        # Add to training data
        self.trainer.train_step(input_ids, target_ids)
    
    def _update_from_feedback(self, feedback: float) -> None:
        """Update model parameters based on user feedback."""
        # This is a simplified feedback mechanism
        # In practice, you would implement a more sophisticated learning algorithm
        if feedback > 0:
            self.temperature = max(0.1, self.temperature - 0.1)
        else:
            self.temperature = min(1.0, self.temperature + 0.1)
    
    def save_state(self, path: str) -> None:
        """Save agent state to disk."""
        state = {
            'model': self.model,
            'memory': self.memory,
            'conversation_history': self.conversation_history,
            'persona': self.persona,
            'learning_rate': self.learning_rate,
            'temperature': self.temperature,
            'top_k': self.top_k
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(state, f)
            self.logger.info(f"Agent state saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving agent state: {str(e)}")
    
    def load_state(self, path: str) -> None:
        """Load agent state from disk."""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.model = state['model']
            self.memory = state['memory']
            self.conversation_history = state['conversation_history']
            self.persona = state['persona']
            self.learning_rate = state['learning_rate']
            self.temperature = state['temperature']
            self.top_k = state['top_k']
            
            self.logger.info(f"Agent state loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading agent state: {str(e)}")
    
    def spawn_sub_agent(self, task: str) -> 'AutonomousAgent':
        """Spawn a sub-agent for a specific task."""
        sub_agent = AutonomousAgent()
        sub_agent.persona['name'] = f"{self.persona['name']}-Sub"
        sub_agent.persona['expertise'] = [task]
        return sub_agent
    
    def debate(self, topic: str, num_sub_agents: int = 2) -> List[Dict]:
        """Initiate a multi-agent debate on a topic."""
        sub_agents = [self.spawn_sub_agent(f"Debate-{i+1}") for i in range(num_sub_agents)]
        debate_history = []
        
        # Each sub-agent takes turns presenting arguments
        for i, agent in enumerate(sub_agents):
            argument = agent.process_input(f"Present your argument about {topic}")
            debate_history.append({
                'agent': agent.persona['name'],
                'argument': argument,
                'timestamp': datetime.now().isoformat()
            })
        
        return debate_history 