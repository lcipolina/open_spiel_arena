# utils/common_utils.py
"""Common utility functions for the OpenSpiel LLM Arena project.

This module provides shared utility functions for logging, configuration,
and other cross-cutting concerns.
"""

import logging

def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger for the simulation.

    Args:
        name: The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def parse_agents(agent_strings: list) -> list:
    """Parse agent configurations from CLI arguments

    Args:
        agent_strings: List of strings in format "type:model" or "type"

    Returns:
        List of agent config dictionaries

    Example:
        ["llm:gpt-4", "random"] →
        [
            {"type": "llm", "model": "gpt-4"},
            {"type": "random", "model": None}
        ]
    """
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
    from tabulate import tabulate

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