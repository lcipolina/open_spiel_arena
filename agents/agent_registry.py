'''
agent_registry.py

Centralized agent registry for dynamic agent loading.
'''

import logging
from agents.llm_registry import LLM_REGISTRY
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.llm_agent import LLMAgent

# Configure logger
logger = logging.getLogger(__name__)

# Register all available agent types
AGENT_REGISTRY = {
    "llm": LLMAgent,
    "random": RandomAgent,
    "human": HumanAgent,
}
