"""
Command-line interface for interacting with the NeuraFlux agent.
"""

import argparse
import sys
from typing import List, Optional
import json
import os
from datetime import datetime
from colorama import init, Fore, Style
import numpy as np

from ..core.transformer import NeuralTransformer
from ..core.memory import HierarchicalMemory
from ..training.trainer import Optimizer, Trainer

class CLI:
    def __init__(self, model_path: Optional[str] = None):
        # Initialize colorama for colored output
        init()
        
        # Initialize model and memory
        self.model = self._initialize_model()
        self.memory = HierarchicalMemory()
        
        # Load model if path provided
        if model_path:
            self._load_model(model_path)
        
        # Command history
        self.history: List[str] = []
        
        # Available commands
        self.commands = {
            "!help": self.show_help,
            "!clear": self.clear_screen,
            "!history": self.show_history,
            "!save": self.save_conversation,
            "!load": self.load_conversation,
            "!web_search": self.web_search,
            "!execute_code": self.execute_code,
            "!adjust_tone": self.adjust_tone,
            "!memory_stats": self.show_memory_stats,
            "!quit": self.quit
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
            with open(model_path, 'r') as f:
                checkpoint = json.load(f)
            
            # Load model parameters
            self.model.token_embedding = np.array(checkpoint["params"]["token_embedding"])
            self.model.pos_embedding = np.array(checkpoint["params"]["pos_embedding"])
            self.model.output_proj = np.array(checkpoint["params"]["output_proj"])
            
            print(f"{Fore.GREEN}Model loaded successfully from {model_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
    
    def run(self) -> None:
        """Run the CLI interface."""
        self.clear_screen()
        print(f"{Fore.CYAN}Welcome to NeuraFlux CLI! Type !help for available commands.{Style.RESET_ALL}")
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
                
                # Check for commands
                if user_input.startswith("!"):
                    cmd = user_input.split()[0]
                    if cmd in self.commands:
                        self.commands[cmd](user_input)
                        continue
                
                # Add to history
                self.history.append(user_input)
                
                # Generate response
                response = self._generate_response(user_input)
                
                # Store in memory
                self.memory.add_memory(user_input)
                self.memory.add_memory(response)
                
                # Print response
                print(f"{Fore.BLUE}NeuraFlux: {Style.RESET_ALL}{response}")
                
            except KeyboardInterrupt:
                print("\n")
                self.quit()
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def _generate_response(self, user_input: str) -> str:
        """Generate a response using the transformer model."""
        # Convert input to token IDs (simplified)
        input_ids = np.array([[hash(word) % self.model.vocab_size for word in user_input.split()]])
        
        # Generate response
        output_ids = self.model.generate(input_ids, max_length=100)
        
        # Convert back to text (simplified)
        return " ".join([str(id) for id in output_ids[0]])
    
    def show_help(self, _: str) -> None:
        """Show available commands."""
        print(f"{Fore.CYAN}Available commands:{Style.RESET_ALL}")
        for cmd in self.commands:
            print(f"  {cmd}")
    
    def clear_screen(self, _: str = "") -> None:
        """Clear the screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_history(self, _: str) -> None:
        """Show command history."""
        for i, cmd in enumerate(self.history, 1):
            print(f"{i}. {cmd}")
    
    def save_conversation(self, _: str) -> None:
        """Save the current conversation to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"{Fore.GREEN}Conversation saved to {filename}{Style.RESET_ALL}")
    
    def load_conversation(self, cmd: str) -> None:
        """Load a conversation from a file."""
        try:
            filename = cmd.split()[1]
            with open(filename, 'r') as f:
                self.history = json.load(f)
            print(f"{Fore.GREEN}Conversation loaded from {filename}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading conversation: {str(e)}{Style.RESET_ALL}")
    
    def web_search(self, cmd: str) -> None:
        """Perform a web search (placeholder)."""
        query = " ".join(cmd.split()[1:])
        print(f"{Fore.YELLOW}Web search for: {query}{Style.RESET_ALL}")
        # Implement web search functionality
    
    def execute_code(self, cmd: str) -> None:
        """Execute code in a safe environment (placeholder)."""
        code = " ".join(cmd.split()[1:])
        print(f"{Fore.YELLOW}Executing code: {code}{Style.RESET_ALL}")
        # Implement safe code execution
    
    def adjust_tone(self, cmd: str) -> None:
        """Adjust the agent's response tone (placeholder)."""
        tone = " ".join(cmd.split()[1:])
        print(f"{Fore.YELLOW}Adjusting tone to: {tone}{Style.RESET_ALL}")
        # Implement tone adjustment
    
    def show_memory_stats(self, _: str) -> None:
        """Show memory statistics."""
        print(f"{Fore.CYAN}Memory Statistics:{Style.RESET_ALL}")
        print(f"Short-term memories: {len(self.memory.short_term.memories)}")
        print(f"Long-term memories: {len(self.memory.long_term.memory_index)}")
    
    def quit(self, _: str = "") -> None:
        """Exit the CLI."""
        print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="NeuraFlux CLI Interface")
    parser.add_argument("--model", type=str, help="Path to a trained model checkpoint")
    args = parser.parse_args()
    
    cli = CLI(args.model)
    cli.run()

if __name__ == "__main__":
    main() 