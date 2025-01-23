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
from games.registry import GAMES_REGISTRY
from configs.configs import default_simulation_config
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent

from games.registry import registry
from envs.open_spiel_env import OpenSpielEnv

from utils.common_utils import parse_agents, print_simulation_summary



def run_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the OpenSpiel simulation given a config dictionary.
    Returns a dictionary with aggregated outcomes or stats.

    Config keys may include:
      - game_name (str): e.g. "tic_tac_toe"
      - rounds (int): how many episodes to play
      - agents (list of dict): each dict has "type": "human"/"llm"/"random_bot", etc.
      - seed (int or None)
      - alternate_first_player (bool)
      - max_game_rounds (int or None): for iterated games.
      ...
    """

    # 1. Set up random seed if specified
    if config.get("seed") is not None:
        random.seed(config["seed"])

    # 2. Load the game from the registry
    try:
        loader = registry.get_game_loader(config["game_name"])
        game = loader()
    except ValueError as e:
        raise RuntimeError(f"Game loading failed: {str(e)}") from e

    # Initialize environment
    env = initialize_environment(game, config)

    # Agent setup
    agents = create_agents(config, env)

    # Run simulation loop
    return run_simulation_loop(env, agents, config)

def initialize_environment(game, config):
    """Initialize game environment"""
    return OpenSpielEnv(
        game=game,
        game_name=registry.get_display_name(config["game_name"]),
        player_type=config["player_types"],   #TODO: see what is this, and whether it should be 'player_typeS'
        max_game_rounds=config.get("max_game_rounds")
    )

def create_agents(config: Dict[str, Any], env: OpenSpielEnv) -> Dict[str, Any]:
    """Create agent instances based on configuration

    Args:
        config: Simulation configuration dictionary
        env: Initialized game environment

    Returns:
        Dictionary of agent instances keyed by player name

    Raises:
        ValueError: For invalid agent types or missing LLM models
    """
    agents = {}

    for idx, agent_cfg in enumerate(config["agents"]):
        agent_type = agent_cfg["type"].lower()
        agent_name = f"Player {idx+1}"

        # Human Agent
        if agent_type == "human":
            from agents.human_agent import HumanAgent
            agents[agent_name] = HumanAgent(
                player_name=agent_name
            )

        # Random Bot
        elif agent_type == "random":
            from agents.random_agent import RandomAgent
            agents[agent_name] = RandomAgent(
                seed=config.get("seed")
            )

        # LLM Agent
        elif agent_type == "llm":
            from agents.llm_agent import LLMAgent

            if not agent_cfg.get("model"):
                raise ValueError(
                    f"LLM agent requires model specification. "
                    f"Missing model for {agent_name}"
                )

            agents[agent_name] = LLMAgent(
                model_name=agent_cfg["model"],
                game=env.game,
                player_id=idx,
                temperature=agent_cfg.get("temperature", 0.7),
                max_tokens=agent_cfg.get("max_tokens", 128)
            )

        # Unknown Agent Type
        else:
            raise ValueError(
                f"Unsupported agent type: '{agent_type}'. "
                f"Valid types: human, random, llm"
            )

    # Validate agent count matches game requirements
    num_players = env.game.num_players()
    if len(agents) != num_players:
        raise ValueError(
            f"Game requires {num_players} players. "
            f"Configured {len(agents)} agents."
        )

    return agents

def run_simulation_loop(env, agents, config):
    """Core simulation execution"""
    results = []
    for episode in range(config["rounds"]):
        observation = env.reset()
        done = False

        while not done:
            current_player = env.current_player()
            agent = agents[f"Player {current_player+1}"]

            action = agent.act(observation)
            observation, reward, done, info = env.step(action)

            if done:
                results.append({
                    "episode": episode,
                    "winner": info.get("winner"),
                    "scores": info.get("scores")
                })

    return {
        "game": config["game_name"],
        "config": config,
        "results": results
    }

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Run OpenSpiel simulations")
    parser.add_argument(
        "-g", "--game",
        required=True,
        help="Name of the game to simulate"
    )
    parser.add_argument(
        "-r", "--rounds",
        type=int,
        default=10,
        help="Number of rounds to simulate"
    )
    parser.add_argument(
        "-a", "--agents",
        nargs="+",
        required=True,
        help="Agent configurations (type:model)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    config = {
        "game_name": args.game,
        "rounds": args.rounds,
        "seed": args.seed,
        "agents": parse_agents(args.agents)
    }

    results = run_simulation(config)
    print_simulation_summary(results)

if __name__ == "__main__":
    main()