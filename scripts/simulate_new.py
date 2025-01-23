#!/usr/bin/env python3
"""
simulate.py

Runs game simulations with configurable agents and tracks outcomes.
Supports both CLI arguments and config dictionaries.
"""

import argparse
import random
import json
import sys
import os
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.open_spiel_env import OpenSpielEnv
from games_registry import GAMES_REGISTRY
from configs import default_simulation_config
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
import pyspiel

def run_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Core simulation logic (unchanged from original)."""
    # ... [Keep the original run_simulation function exactly as provided] ...

def cli_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert CLI arguments to config dictionary."""
    config = default_simulation_config()

    # Override config with CLI arguments
    config.update({
        "game_name": args.game,
        "rounds": args.rounds,
        "max_game_rounds": args.max_game_rounds,
        "alternate_first_player": args.alternate_first,
        "agents": []
    })

    # Handle player types and models
    for i, (p_type, p_model) in enumerate(zip(args.player_types, args.player_models)):
        config["agents"].append({
            "type": p_type,
            "name": f"Player {i+1}",
            "model": p_model if p_model else None
        })

    return config

def main():
    """Command-line interface with argparse."""
    parser = argparse.ArgumentParser(description="Run OpenSpiel simulations")
    parser.add_argument("--game", default="tic_tac_toe", choices=list(GAMES_REGISTRY.keys()))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-game-rounds", type=int)
    parser.add_argument("--player-types", nargs="+", default=["human", "random_bot"],
                        choices=["human", "random_bot", "llm"])
    parser.add_argument("--player-models", nargs="+", default=[])
    parser.add_argument("--alternate-first", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config-file", help="JSON config file to override defaults")

    args = parser.parse_args()

    # Load config file if specified
    if args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)
    else:
        config = cli_to_config(args)

    # Run simulation and display results
    results = run_simulation(config)

    # Generate leaderboard
    leaderboard: Dict[str, int] = {}
    for outcome in results.get("outcomes", []):
        winner = outcome.get("winner")
        if winner and winner != "tie":
            leaderboard[winner] = leaderboard.get(winner, 0) + 1

    print("\n=== Final Leaderboard ===")
    for player, wins in sorted(leaderboard.items(), key=lambda x: -x[1]):
        print(f"{player}: {wins} wins")

if __name__ == "__main__":
    main()