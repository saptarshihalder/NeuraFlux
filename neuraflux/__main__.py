"""
Main entry point for the NeuraFlux application.
"""

import argparse
import sys
import os
import logging
from datetime import datetime

from .interface.cli import CLI
from .interface.web_ui import WebUI
from .core.agent import AutonomousAgent

def setup_logging():
    """Configure logging for the application."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"neuraflux_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point for the NeuraFlux application."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="NeuraFlux: Autonomous LLM Agent")
    parser.add_argument("--model", type=str, help="Path to a trained model checkpoint")
    parser.add_argument("--state", type=str, help="Path to a saved agent state")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--ui", choices=["cli", "web"], default="web", help="User interface mode")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize agent
        agent = AutonomousAgent(model_path=args.model)
        
        # Load state if provided
        if args.state:
            agent.load_state(args.state)
        
        # Run appropriate interface
        if args.ui == "cli":
            cli = CLI(agent)
            cli.run()
        else:
            ui = WebUI(agent)
            ui.run()
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 