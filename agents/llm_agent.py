"""
llm_agent.py

Implements an agent that queries an LLM for its action.
"""

import logging
import random
import os, json
import ray
from typing import Any, Dict, List, Optional
from agents.llm_registry import LLM_REGISTRY
from vllm import SamplingParams
from .base_agent import BaseAgent

# Get values from SLURM (default if not found)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 250))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))  # the lower the more deterministic

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

        # Loads the LLM model to GPU  - TODO: ver como se hace esto con litellm
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
        prompt = observation.get("prompt", None)

        # Call batch function (use Ray if initialized, otherwise call directly for debugging)
        if ray.is_initialized():
            action_dict = ray.get(batch_llm_decide_moves.remote(
                {0: self.model_name},
                {0: prompt}
            ))
        else:
            action_dict = batch_llm_decide_moves(
                {0: self.model_name},
                {0: prompt}
            )

        chosen_action = action_dict[0]["action"]
        reasoning = action_dict[0].get("reasoning", "N/A")
        return {"action": chosen_action, "reasoning": reasoning} #TODO: should be a dictionary for the other agents too!

    #TODO: uncomment this!
#@ray.remote  # The function will be a separate Ray task or actor
def batch_llm_decide_moves(
    model_names: Dict[int, str],  # Supports multiple LLMs per player
    prompts: Dict[int, str]
) -> Dict[int, int]:
    """
    Queries vLLM in batch mode to decide moves for multiple players, supporting multiple LLM models.

    Args:
        model_names (Dict[int, str]): Dictionary mapping player IDs to their respective LLM model names.
        prompts (Dict[int, str]): Dictionary mapping player IDs to prompts.
        legal_actions (Dict[int, tuple]): Dictionary mapping player IDs to legal actions.

    Returns:
        Dict[int, int]: Mapping of player ID to chosen action.
    """

    # Load all models in use
    llm_instances = {
        player_id: LLM_REGISTRY[model_name]["model_loader"]()
        for player_id, model_name in model_names.items()
    }
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    # Run batch inference for each LLM model separately
    actions_dict = {}
    for player_id, llm in llm_instances.items():
        response = llm.generate([prompts[player_id]], sampling_params)[0]  # Single response

        # Extract action from response
        response_text = response.outputs[0].text  # Assuming this is a string
        parsed_response = json.loads(response_text.replace("'", '"'))  # Convert to valid JSON

        actions_dict[player_id] = {
                'action': parsed_response.get('action'),
                'reasoning': parsed_response.get('reasoning')
            }

    return actions_dict  # dict[playerID, {chosen action, reasoning}]
