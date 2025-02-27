# utils/llm_utils.py
"""Utility functions for Large Language Model (LLM) integration.

This module provides helper functions to generate prompts and interact with LLMs
for decision-making in game simulations.
"""

import os
from typing import List, Optional, Dict
import random
from vllm import SamplingParams
from agents.llm_registry import LLM_REGISTRY
import ray
import contextlib

import gc
import torch
from vllm.distributed import (
    destroy_model_parallel,
    destroy_distributed_environment
)

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


def cleanup_vllm(llm=None):
    """Properly cleans GPU memory before loading a new model."""
    print("🧹 Cleaning up vLLM model from GPU memory...")

    destroy_model_parallel()
    destroy_distributed_environment()

    if llm:
        del llm.llm_engine.model_executor
        del llm

    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("cleanup_vllm: GPU memory successfully freed.")

def close_simulation():
    """Cleans up all LLMs & GPU memory after the game ends."""
    global CURRENT_LLM

    print(" Closing simulation: Clearing all LLMs & GPU memory...")

    # Ensure the currently loaded LLM is cleaned up
    if CURRENT_LLM is not None:
        cleanup_vllm(CURRENT_LLM)
        CURRENT_LLM = None

    # Destroy parallel processing (if any active model is still in memory)
    destroy_model_parallel()
    destroy_distributed_environment()

    # Ensure all LLM instances are deleted (if used in batch mode)
    if "llm_instances" in globals():
        for llm in llm_instances.values():
            del llm
        del llm_instances

    # Force garbage collection
    gc.collect()  # Run garbage collection

    # Free GPU memory
    torch.cuda.empty_cache() # Free unused memory
    torch.cuda.synchronize() # Ensure it's cleared

    print("Simulation closed successfully. GPU memory freed.")
