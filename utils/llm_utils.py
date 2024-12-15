# utils/llm_utils.py
"""Utility functions for Large Language Model (LLM) integration.

This module provides helper functions to generate prompts and interact with LLMs
for decision-making in game simulations.
"""

from functools import lru_cache
from typing import List, Any


def generate_prompt(game_name: str, state: str, legal_actions: List[int]) -> str:
    """Generate a natural language prompt for the LLM to decide the next move.

    Args:
        game_name: The name of the game.
        state: The current game state as a string.
        legal_actions: The list of legal actions available to the player.

    Returns:
        str: A prompt string for the LLM.
    """
    return (
        f"Game: {game_name}\n"
        f"State:\n{state}\n"
        f"Legal actions: {legal_actions}\n"
        "Choose the next action (provide the action number)."
    )

@lru_cache(maxsize=128)
def llm_decide_move(llm: Any, prompt: str, legal_actions: List[int]) -> int:
    """Use an LLM to decide the next move, with caching for repeated prompts.

    Args:
        llm: The LLM pipeline instance (e.g., from Hugging Face).
        prompt: The prompt string provided to the LLM.
        legal_actions: The list of legal actions available.

    Returns:
        int: The action selected by the LLM.
    """
    response = llm(prompt, max_new_tokens=30, pad_token_id=50256)[0]["generated_text"]
    for word in response.split():
        try:
            move = int(word)
            if move in legal_actions:
                return move
        except ValueError:
            continue
    return legal_actions[0]  # Fallback to the first legal action if no valid move is found


