# utils/common_utils.py
"""Common utility functions for the OpenSpiel LLM Arena project.

This module provides shared utility functions for logging, configuration,
and other cross-cutting concerns.
"""

import logging
from tabulate import tabulate
from games.registry import registry
from typing import Dict, Any




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


#TODO: (lck) see if this needs to be removed or check for other config entries
def validate_config(config: Dict[str, Any]) -> None:
    """Validates the configuration."""
    game_name = config["env_config"]["game_name"]
    num_players = registry.get_game_loader(game_name)().num_players()

    if len(config["agents"]) != num_players:
        raise ValueError(
            f"Game '{game_name}' requires {num_players} players, "
            f"but {len(config['agents'])} agents were provided."
        )
