"""
human_agent.py

Implements an agent that asks the user for input to choose an action.
"""

from typing import List, Any
from .base_agent import BaseAgent

class HumanAgent(BaseAgent):
    """
    An agent that queries the user for which action to pick.
    """

    def __init__(self, game_name: str):
        """
        Args:
            game_name (str): A human-readable name for the game, used for prompting.
        """
        self.game_name = game_name

    def compute_action(self, legal_actions: List[int], state: Any) -> int:
        """
        Prompts the user for a legal move.

        Args:
            legal_actions (List[int]): The set of legal actions for the current player.
            state (Any): The current OpenSpiel state.

        Returns:
            int: The chosen action.
        """
        print(f"Current state of {self.game_name}:\n{state}")
        print(f"Your options: {legal_actions}")
        while True:
            try:
                action_str = input("Enter your action (number): ")
                action = int(action_str)
                if action in legal_actions:
                    return action
            except ValueError:
                pass
            print("Invalid action. Please choose from:", legal_actions)
