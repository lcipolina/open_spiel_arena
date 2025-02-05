# utils/llm_utils.py
"""Utility functions for Large Language Model (LLM) integration.

This module provides helper functions to generate prompts and interact with LLMs
for decision-making in game simulations.
"""

from functools import lru_cache
from typing import List, Any, Optional
import random


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
    info_text = f"{info}\n" if info else ""  # ✅ Separate conditional string assignment

    return (
        f"You are playing the Game: {game_name}\n"
        f"State:\n{state}\n"
        f"Legal actions: {legal_actions}\n"
        f"{info_text}"
        "Your task is to choose the next action. Provide only the number of "
        "your next move from the list of legal actions. Do not provide any additional text or explanation."
    )




@lru_cache(maxsize=128)
def llm_decide_move(llm: Any, prompt: str, legal_actions: tuple) -> int:
    """Use an LLM to decide the next move, with caching for repeated prompts.

    Args:
        llm: The LLM pipeline instance (e.g., from Hugging Face).
        prompt: The prompt string provided to the LLM.
        legal_actions: The list of legal actions available (converted to tuple).

    Returns:
        int: The action selected by the LLM.
    """

    # TODO(lkun): test this: temperature = 0.1 #less creative

    response = llm(prompt, max_new_tokens=30, pad_token_id=50256)[0]["generated_text"]
    for word in response.split():
         try:
             move = int(word)
             if move in legal_actions:  # Validate the move against legal actions
                 return move
         except ValueError:
             continue

    return random.choice(legal_actions)   # Fallback if no valid move is found
