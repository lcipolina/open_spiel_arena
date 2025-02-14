# utils/common_utils.py
"""Common utility functions for the OpenSpiel LLM Arena project.

This module provides shared utility functions for logging, configuration,
and other cross-cutting concerns.
"""

import os
import json
import logging
from tabulate import tabulate
from games.registry import registry
from typing import Dict, Any, List


#TODO: there 3 functions are not used but useful! it contains the state history!

def save_results(game_name: str, final_scores: List[float], state: Any):
    results = prepare_results(game_name, final_scores, state)
    filename = get_results_filename(game_name)
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

def prepare_results(game_name: str, final_scores: List[float], state: Any) -> Dict[str, Any]:
    return {
        "game_name": game_name,
        "final_state": str(state),
        "returns": list(final_scores),  # Ensure it's JSON-serializable
        "history": state.history_str(),
    }

def get_results_filename(game_name: str) -> str:
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"{game_name.lower().replace(' ', '_')}_results.json")


def print_total_scores(game_name: str, summary: Dict[str, Any]):
    """
    Prints the total scores summary for a game.

    Args:
        game_name: The name of the game being summarized.
        summary: The dictionary containing the game summary.
    """

    print(f"\nTotal scores across all episodes for game {game_name}:")

    # 🔹 Ensure we loop through actual player summaries
    if game_name in summary:
        print(f"Game summary: {summary[game_name]}")
    else:
        for player_id, stats in summary.items():
            print(f"Player {player_id}: {stats}")
