"""
Web-based UI for the NeuraFlux chatbot using Streamlit.
"""

import streamlit as st
from streamlit_chat import message
import os
from typing import List, Dict
import json
from datetime import datetime

from ..core.agent import AutonomousAgent

class WebUI:
    def __init__(self, agent: AutonomousAgent):
        self.agent = agent
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'model_path' not in st.session_state:
            st.session_state.model_path = None
        if 'state_path' not in st.session_state:
            st.session_state.state_path = None
    
    def render_sidebar(self):
        """Render the sidebar with settings and controls."""
        with st.sidebar:
            st.title("NeuraFlux Settings")
            
            # Model settings
            st.subheader("Model Settings")
            st.session_state.model_path = st.text_input(
                "Model Checkpoint Path",
                value=st.session_state.model_path or "",
                help="Path to a trained model checkpoint"
            )
            
            st.session_state.state_path = st.text_input(
                "Agent State Path",
                value=st.session_state.state_path or "",
                help="Path to a saved agent state"
            )
            
            if st.button("Load Model/State"):
                if st.session_state.model_path:
                    self.agent._load_model(st.session_state.model_path)
                    st.success("Model loaded successfully!")
                if st.session_state.state_path:
                    self.agent.load_state(st.session_state.state_path)
                    st.success("Agent state loaded successfully!")
            
            # Memory settings
            st.subheader("Memory Settings")
            if st.button("Clear Memory"):
                self.agent.memory.clear_short_term()
                st.success("Short-term memory cleared!")
            
            # Export chat history
            st.subheader("Export")
            if st.button("Export Chat History"):
                self.export_chat_history()
            
            # Agent persona
            st.subheader("Agent Persona")
            st.write(f"Name: {self.agent.persona['name']}")
            st.write("Traits:", ", ".join(self.agent.persona['traits']))
            st.write("Expertise:", ", ".join(self.agent.persona['expertise']))
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        st.title("NeuraFlux Chat")
        
        # Display chat messages
        for i, msg in enumerate(st.session_state.messages):
            message(
                msg["content"],
                key=str(i),
                is_user=msg["role"] == "user",
                avatar_style="big-smile" if msg["role"] == "user" else "bottts"
            )
        
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response
            with st.spinner("Thinking..."):
                response = self.agent.process_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update the chat display
            st.rerun()
    
    def export_chat_history(self):
        """Export chat history to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(st.session_state.messages, f, indent=2)
        
        st.success(f"Chat history exported to {filename}")
    
    def run(self):
        """Run the web UI."""
        st.set_page_config(
            page_title="NeuraFlux Chat",
            page_icon="🤖",
            layout="wide"
        )
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main chat interface
        self.render_chat_interface()

def main():
    """Main entry point for the web UI."""
    # Initialize agent
    agent = AutonomousAgent()
    
    # Create and run UI
    ui = WebUI(agent)
    ui.run()

if __name__ == "__main__":
    main() 