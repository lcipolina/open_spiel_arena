"""
random_agent.py

Implements an agent that picks a random legal action.
"""

import random
from typing import List, Any
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Agent that selects an action uniformly at random from the legal actions.
    """

    def __init__(self, seed: int = None):
        """
        Args:
            llm (Any): The LLM instance (e.g., an OpenAI API wrapper, or any callable).
            game_name (str): The game's name for context in the prompt.
        """
        self.random_generator = random.Random(seed)


    def compute_action(self, legal_actions: List[int], state: Any) -> int:
        """
        Randomly picks a legal action.

        Args:
            legal_actions (List[int]): The set of legal actions for the current player.
            state (Any): The current OpenSpiel state (not used).

        Returns:
            int: A randomly selected action.
        """
        return self.random_generator.choice(legal_actions)
