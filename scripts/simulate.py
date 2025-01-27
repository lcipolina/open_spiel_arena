#!/usr/bin/env python3
"""
simulate.py

Runs game simulations with configurable agents and tracks outcomes.
Supports both CLI arguments and config dictionaries.
"""

import os, sys
import logging
import random
from typing import Dict, Any

from configs.configs import build_cli_parser, parse_config
from envs.open_spiel_env import OpenSpielEnv
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from games.registry import registry # Initilizes an empty registry dictionary
from games import loaders  # Adds the games to the registry dictionary
from utils.common_utils import print_simulation_summary, validate_config



def initialize_environment(game, config: Dict[str, Any]) -> OpenSpielEnv:
    """Initializes the game environment."""
    player_types = [agent["type"] for agent in config["agents"]]
    return OpenSpielEnv(
        game=game,
        game_name=config["env_config"]["game_name"],
        player_type=player_types,  # TODO: check in OpenSpielEnv whether we need to pass one or many
        max_game_rounds=config["env_config"].get("max_game_rounds"),
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
            agents[agent_name] = HumanAgent(
                player_name=agent_name
            )

        # Random Bot
        elif agent_type == "random":
            agents[agent_name] = RandomAgent(
                seed=config.get("seed")
            )

        # LLM Agent
        elif agent_type == "llm":
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


def run_simulation(args) -> Dict[str, Any]:
    """
    Orchestrates the simulation workflow.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dict: Simulation results.
    """

    # Parse and validate game's configuration
    config = parse_config(args)
    validate_config(config)

    # Set up logging
    logging.basicConfig(level=getattr(logging, config["log_level"].upper()))
    logger = logging.getLogger(__name__)
    logger.info("Starting simulation...")


    # 1. Set up random seed if specified
    if config.get("seed") is not None:
        random.seed(config["seed"])

    # TODO: FOR NOW THE GAME SIMULATOR IS NOT USED
    # 2. Load the pyspiel game object
    try:
        loader = registry.get_game_loader(config["env_config"]["game_name"])
        game = loader()
    except ValueError as e:
        raise RuntimeError(f"Game loading failed: {str(e)}") from e

    # Initialize environment
    env = initialize_environment(game, config)

    # Agent setup
    agents = create_agents(config, env)

    # Run simulation loop
    return run_simulation_loop(env, agents, config)


def run_simulation_loop(env, agents, config):
    """Core simulation execution"""
    results = []
    for episode in range(config["num_episodes"]):
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

    # Build the CLI parser
    parser = build_cli_parser()
    args = parser.parse_args()

    # Run the simulation
    results = run_simulation(args)
    print_simulation_summary(results)


if __name__ == "__main__":
    main()
