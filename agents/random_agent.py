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

    def compute_action(self, legal_actions: List[int], state: Any) -> int:
        """
        Randomly picks a legal action.

        Args:
            legal_actions (List[int]): The set of legal actions for the current player.
            state (Any): The current OpenSpiel state (not used).

        Returns:
            int: A randomly selected action.
        """
        return random.choice(legal_actions)
