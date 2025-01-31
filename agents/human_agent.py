"""
human_agent.py

Implements an agent that asks the user for input to choose an action.
"""

from typing import Any, Dict
from .base_agent import BaseAgent

from agents.llm_utils import generate_prompt


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

    def compute_action(self, observation: Dict[str,Any]) -> int:
        """
        Prompts the user for a legal move.

        Args:
            legal_actions (List[int]): The set of legal actions for the current player.
            state (Any): The current OpenSpiel state.

        Returns:
            int: The chosen action.
        """
        legal_actions=observation["legal_actions"]
        state=observation.get("state_string")
        info = observation.get("info",None)
        prompt= observation.get("prompt",None)

        # Same prompt as the LLM agent
        if prompt is None:
           prompt = generate_prompt(self.game_name, str(state), legal_actions, info = info)

        print(prompt)
        while True:
            try:
                action_str = input("Enter your action (number): ")
                action = int(action_str)
                if action in legal_actions:
                    return action
            except ValueError:
                pass
            print("Invalid action. Please choose from:", legal_actions)
