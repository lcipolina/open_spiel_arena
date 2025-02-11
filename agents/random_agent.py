"""
random_agent.py

Implements an agent that selects a random action.
"""

import random
from typing import Dict, Any
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Agent that selects an action uniformly at random from the legal actions.
    """

    def __init__(self, seed: int = None, *args, **kwargs):
        """
        Args:
            llm (Any): The LLM instance (e.g., an OpenAI API wrapper, or any callable).
            game_name (str): The game's name for context in the prompt.
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.
         """
        super().__init__(agent_type="random")
        self.random_generator = random.Random(seed)

    def compute_action(self, observation: Dict[str,Any]) -> int:
        """
        Randomly picks a legal action.

        Args:
            legal_actions (List[int]): The set of legal actions for the current player.
            *args: Unused additional arguments for consistency.
            **kwargs: Unused keyword arguments (e.g., state, info).

        Returns:
            int: A randomly selected action.
        """
        return self.random_generator.choice(observation["legal_actions"])
