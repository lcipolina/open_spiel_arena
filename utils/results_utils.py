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


#TODO: delete this function
'''
def parse_agents(agent_strings: list) -> list:
    """Convert 'type:model' strings to agent configs"""
    agents = []
    for agent_str in agent_strings:
        if ":" in agent_str:
            agent_type, model = agent_str.split(":", 1)
        else:
            agent_type = agent_str
            model = None

        agents.append({
            "type": agent_type.strip().lower(),
            "model": model.strip() if model else None
        })
    return agents
'''

def print_total_scores(game_name,total_scores):
    """
    Print total scores for all players in a readable format.

    Args:
        total_scores: A dictionary where keys are player IDs and values are
                      their total scores across episodes.
    """
    print(f"Total scores across all episodes for game {game_name}:")
    for player, score in total_scores.items():
        print(f"Player {player}: {score}")


#TODO: delete this function
def print_simulation_summary(results: dict):
    """Print formatted simulation results"""

    # Basic stats
    print(f"\n{' Simulation Summary ':=^50}")
    print(f"Game: {results['game']}")
    print(f"Rounds: {results['config']['rounds']}")

    # Leaderboard
    leaderboard = {}
    ties = 0

    for result in results["results"]:
        if result["winner"]:
            leaderboard[result["winner"]] = leaderboard.get(result["winner"], 0) + 1
        else:
            ties += 1

    # Prepare table data
    table = []
    for player, wins in leaderboard.items():
        table.append([
            player,
            wins,
            f"{(wins/len(results['results']))*100:.1f}%"
        ])

    if ties:
        table.append([
            "Ties",
            ties,
            f"{(ties/len(results['results']))*100:.1f}%"
        ])

    # Print formatted table
    print("\nLeaderboard:")
    print(tabulate(
        table,
        headers=["Player", "Wins", "Win Rate"],
        tablefmt="rounded_outline"
    ))

    # Detailed stats
    print("\nDetailed Results:")
    for i, result in enumerate(results["results"], 1):
        print(f"Round {i}: ", end="")
        if result["winner"]:
            print(f"Winner: {result['winner']} (Scores: {result['scores']})")
        else:
            print(f"Tie (Scores: {result['scores']})")


# TODO: not sure if these ones are needed

def _initialize_outcomes(self) -> Dict[str, Any]:
        """Initializes an outcomes dictionary to track wins, losses, ties, etc."""
        return {
            "wins": {name: 0 for name in self.player_types.keys()},
            "losses": {name: 0 for name in self.player_types.keys()},
            "ties": 0
        }

def record_outcomes(self, final_scores: List[float], outcomes: Dict[str, Any]) -> str:
        """Records the outcome of a single game round.

        Args:
            final_scores (List[float]): Final cumulative scores of all players.
            outcomes (Dict[str, Any]): Dictionary to record wins, losses, and ties.

        Returns:
            str: Name of the winner or "tie" if there is no single winner.
        """
        # Check if all scores are equal (a tie)
        if all(score == final_scores[0] for score in final_scores):
            outcomes["ties"] += 1
            return "tie"

        # Find the maximum score and determine winners #TODO (lck: look into this -this is a bit confusing)
        max_score = max(final_scores)
        # Assume players in order "Player 1", "Player 2", etc.
        # This depends on the self.player_type keys (which must be in a stable order)
        # Identify winners/losers by mapping i -> player name
        # Suppose we match indexes to the order of self.llms.keys(), or define your own order #TODO: (lck: look into this)
        sorted_players = sorted(self.player_types.keys())  # or track your own ordering
        winners = [name for i, name in enumerate(sorted_players) if final_scores[i] == max_score]
        losers = [name for i, name in enumerate(sorted_players) if final_scores[i] != max_score]

        if len(winners) == 1:
            outcomes["wins"][winners[0]] += 1
            for loser in losers:
                outcomes["losses"][loser] += 1
            return winners[0]
        else:
            outcomes["ties"] += 1
            return "tie"
