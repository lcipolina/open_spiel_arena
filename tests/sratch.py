# llm_decide_move

import subprocess
import random
from functools import lru_cache
from typing import List, Any, Optional

def generate_prompt(game_name: str, state: str, legal_actions: List[int], info: Optional[str] = None) -> str:
    """Generate a natural language prompt for the LLM to decide the next move."""
    info_text = f"{info}\n" if info else ""
    return (
        f"You are playing the Game: {game_name}\n"
        f"State:\n{state}\n"
        f"Legal actions: {legal_actions}\n"
        f"{info_text}"
        "Your task is to choose the next action. Provide only the number of "
        "your next move from the list of legal actions. Do not provide any additional text or explanation."
    )

@lru_cache(maxsize=128)
def llm_decide_move(prompt: str, legal_actions: tuple) -> int:
    """Use vLLM to decide the next move.

    Args:
        prompt: The prompt string for the LLM.
        legal_actions: The list of legal actions available.

    Returns:
        int: The action selected by the LLM.
    """
    cmd = ["python3", "inference.py", "--prompt", prompt]

    try:
        process = subprocess.run(cmd, text=True, capture_output=True, check=True)
        response = process.stdout.strip()
        for word in response.split():
            try:
                move = int(word)
                if move in legal_actions:
                    return move
            except ValueError:
                continue
    except subprocess.CalledProcessError as e:
        print(f"Error calling LLM: {e}")

    return random.choice(legal_actions)  # Fallback if no valid move is found
