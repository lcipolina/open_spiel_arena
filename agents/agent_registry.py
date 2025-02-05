'''
agent_registry.py

Centralized agent registry for dynamic agent loading.
'''

from typing import Type, Dict, Any
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from agents.base_agent import BaseAgent

# Centralized agent registry
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "human": HumanAgent,
    "random": RandomAgent,
    "llm": LLMAgent,
}

def register_agent(agent_name: str, agent_class: Type[BaseAgent]):
    """Registers a new agent type dynamically."""
    if agent_name in AGENT_REGISTRY:
        raise ValueError(f"Agent type '{agent_name}' is already registered.")
    AGENT_REGISTRY[agent_name] = agent_class
