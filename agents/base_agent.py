"""
base_agent.py

Defines a base class for all agents, which can then be subclassed by
HumanAgent, RandomAgent, LLMAgent, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Any

class BaseAgent(ABC):
    """Abstract base class for agents that pick actions in an OpenSpiel environment."""

    @abstractmethod
    def compute_action(self, legal_actions: List[int], state: Any) -> int:
        """
        Select an action from the list of legal actions.

        Args:
            legal_actions (List[int]): The set of legal actions for the current player.
            state (Any): The current OpenSpiel state (if needed for policy decisions).

        Returns:
            int: The action chosen by the agent.
        """
        pass
