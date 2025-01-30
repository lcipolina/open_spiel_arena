"""
llm_agent.py

Implements an agent that uses an LLM to decide the next move.
"""

from typing import Any, List, Optional
from .base_agent import BaseAgent
from agents.llm_utils import generate_prompt, llm_decide_move

class LLMAgent(BaseAgent):
    """
    Agent that queries a language model (LLM) to pick an action.
    """

    def __init__(self, llm, game_name: str):
        """
        Args:
            llm (Any): The LLM instance (e.g., an OpenAI API wrapper, or any callable).
            game_name (str): The game's name for context in the prompt.
        """
        self.llm = llm
        self.game_name = game_name

    def compute_action(self, legal_actions: List[int], state: Any, info: Optional[str] = None) -> int:
        """
        Uses the LLM to select an action from the legal actions.

        Args:
            legal_actions (List[int]): The set of legal actions for the current player.
            state (Any): The current OpenSpiel state.
            info (Optional[str]): Additional information to include in the prompt.

        Returns:
            int: The action chosen by the LLM.
        """

        # HERE NEEDS TO CALL THE SPECIFIC prompt of the game!!

        prompt = generate_prompt(self.game_name, str(state), legal_actions, info = info)
        return llm_decide_move(self.llm, prompt, tuple(legal_actions))