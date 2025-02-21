"""
llm_agent.py

Implements an agent that queries an LLM for its action.
"""

import logging
import random
import ray
from typing import Any, Dict
from agents.llm_registry import LLM_REGISTRY
from agents.llm_utils import generate_prompt, batch_llm_decide_moves
from .base_agent import BaseAgent


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
        self.model_name = model_name

    def compute_action(self, observation: Dict[str, Any]) -> int:
        """
        Uses the LLM to select an action from the legal actions.

        Args:
            observation (Dict[str, Any]): The observation dictionary containing:
                - legal_actions: The set of legal actions for the current player.
                - state_string: The current OpenSpiel state.
                - info: Additional information to include in the prompt.

        Returns:
            int: The action chosen by the LLM.
        """
        legal_actions = observation["legal_actions"]
        #state = observation.get("state_string")
        #info = observation.get("info", None)
        prompt = observation.get("prompt", None) # TODO: check if the wording is coming correctly

        # If there are no legal actions, return a safe fallback move
        #if not legal_actions:
        #    logging.error(f"LLMAgent for {self.game_name} encountered an empty legal_actions list.")
        #    return -1  # Safe fallback action (invalid but won't break the game)

        #prompt = generate_prompt(self.game_name, str(state), legal_actions, info=info)

        # Call batch function (use Ray if initialized, otherwise call directly for debugging)
        if ray.is_initialized():
            action_dict = ray.get(batch_llm_decide_moves.remote(
                {0: self.model_name},
                {0: prompt},
                {0: tuple(legal_actions)}
            ))
        else:
            action_dict = batch_llm_decide_moves(
                {0: self.model_name},
                {0: prompt},
                {0: tuple(legal_actions)}
            )

        # Extract the single action
        chosen_action = action_dict.get(0, None)

        # If LLM fails to return a valid move, pick randomly
        if chosen_action not in legal_actions:
            logging.warning(f"LLM returned an invalid move: {chosen_action}. Choosing randomly.")
            chosen_action = random.choice(legal_actions)

        return chosen_action
