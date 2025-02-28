# utils/llm_utils.py
"""Utility functions for Large Language Model (LLM) integration.

This module provides helper functions to generate prompts and interact with LLMs
for decision-making in game simulations.
"""

import os
from typing import List, Optional, Dict
import random
from vllm import SamplingParams
import ray
import contextlib

import gc
import torch


from agents.llm_registry import initialize_llm_registry

initialize_llm_registry()
from agents.llm_registry import LLM_REGISTRY


# Set environment variable to allow PyTorch to dynamically allocate more memory on GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Get values from SLURM (default if not found)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 10))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

def generate_prompt(game_name: str,
                    state: str,
                    legal_actions: List[int],
                    info: Optional[str] = None) -> str:
    """Generate a natural language prompt for the LLM to decide the next move.

    Args:
        game_name (str): The name of the game.
        state (str): The current game state as a string.
        legal_actions (List[int]): The list of legal actions available to the player.
        info (Optional[str]): Additional information to include in the prompt (optional).

    Returns:
        str: A prompt string for the LLM.
    """
    info_text = f"{info}\n" if info else ""

    return (
        f"You are playing the Game: {game_name}\n"
        f"State:\n{state}\n"
        f"Legal actions: {legal_actions}\n"
        f"{info_text}"
        "Your task is to choose the next action. Provide only the number of "
        "your next move from the list of legal actions. Do not provide any additional text or explanation."
    )

#TODO: uncomment this!
#@ray.remote  # The function will be a separate Ray task or actor
def batch_llm_decide_moves(
    model_names: Dict[int, str],  # Supports multiple LLMs per player
    prompts: Dict[int, str],
    legal_actions: Dict[int, tuple]
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
    actions = {}
    for player_id, llm in llm_instances.items():
        response = llm.generate([prompts[player_id]], sampling_params)[0]  # Single response

        # Extract action from response
        move = None
        for word in response.outputs[0].text.split():
            try:
                move = int(word)
                if move in legal_actions[player_id]:  # Validate move
                    actions[player_id] = move
                    break
            except ValueError:
                continue

        # If LLM fails to return a valid move, pick randomly  #TODO: this needs to go to the 'invalid action selection' counter
        if player_id not in actions:
            actions[player_id] = random.choice(legal_actions[player_id])

    return actions  # dict[playerID, chosen action]
