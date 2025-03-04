# utils/llm_utils.py
"""Utility functions for Large Language Model (LLM) integration.

This module provides helper functions to generate prompts and interact with LLMs
for decision-making in game simulations.
"""

import os
import json
from typing import List, Optional
from transformers import AutoTokenizer, pipeline

from agents.llm_registry import initialize_llm_registry

initialize_llm_registry() #TODO: I don't kike this design!
from agents.llm_registry import LLM_REGISTRY

#TODO: all this goes into the SLURM later! delete!
# Set environment variable to allow PyTorch to dynamically allocate more memory on GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def format_prompt(input_text:str)->str:
    """Formats the input prompt using Hugging Face's chat template function.

    Args:
        input_text (str): The game prompt.

    Returns:
        str: The correctly formatted prompt for the model.
    """
    messages = [{"role": "user", "content": input_text}]

    # TODO: construir esto con el model name nomas y el model path del OS
    model_path = "/p/data1/mmlaion/marianna/models/google/codegemma-7b-it"

    # Format using apply_chat_template
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted_prompt  # This is passed to vLLM



def generate_prompt_old(game_name: str,
                    state: str,
                    legal_actions: List[int],
                    info: Optional[str] = None) -> str:
        """Generate a structured JSON prompt for an LLM to decide the next move.

        Args:
            game_name (str): The name of the game.
            state (str): The current game state as a string.
            legal_actions (List[int]): The list of legal actions available to the player.
            info (Optional[str]): Additional contextual information.

        Returns:
            str: A JSON-structured prompt.
        """
        action_numbers = [action_dict["action"] for action_dict in legal_actions]

        prompt_dict = {
            "game": game_name,
            "state": state,
            "legal_actions": legal_actions,
            "info": info if info else "",
            "instruction": (f"Your task is to choose the next action. Reply only with a JSON object like "
                f"{{'action': {legal_actions[0]['action']}}} where the value must be one of {action_numbers}.")
        }
        return json.dumps(prompt_dict, indent=2)
