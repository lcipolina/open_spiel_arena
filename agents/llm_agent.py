"""
llm_agent.py

Implements an agent that queries an LLM for its action.
"""

from typing import Any, Dict
from .base_agent import BaseAgent
from agents.llm_registry import LLM_REGISTRY
from agents.llm_utils import generate_prompt, llm_decide_move

class LLMAgent(BaseAgent):
    """
    Agent that queries a language model (LLM) to pick an action.
    """

    def __init__(self, model_name: str, game_name: str):
        """
        Args:
            model_name (str): The name of the LLM to use (from the LLM registry).
            game_name (str): The game's name for context in the prompt.
        """
        super().__init__(agent_type="llm")
        self.game_name = game_name

        # Load the LLM from the registry
        if model_name not in LLM_REGISTRY:
            raise ValueError(f"LLM '{model_name}' is not registered.")

        self.llm = LLM_REGISTRY[model_name]["model_loader"]()

    def compute_action(self, observation: Dict[str,Any]) -> int:
        """
        Uses the LLM to select an action from the legal actions.

        Args:
            observation (Dict[str,Any]): The observation dictionary containing:
                - legal_actions: The set of legal actions for the current player.
                - state_string: The current OpenSpiel state.
                - info: Additional information to include in the prompt.
                - prompt: The prompt for the LLM (optionally coming from some games).
        Returns:
            int: The action chosen by the LLM.
        """
        legal_actions = observation["legal_actions"]
        state = observation.get("state_string")
        info = observation.get("info",None)
        prompt= observation.get("prompt",None)

        if prompt is None:
           prompt = generate_prompt(self.game_name, str(state), legal_actions, info = info)
        return llm_decide_move(self.llm, prompt, tuple(legal_actions))