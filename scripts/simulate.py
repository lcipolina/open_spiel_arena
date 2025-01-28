#!/usr/bin/env python3
"""
simulate.py

Runs game simulations with configurable agents and tracks outcomes.
Supports both CLI arguments and config dictionaries.
"""

import logging
import random
from typing import Dict, Any, List

from configs.configs import build_cli_parser, parse_config
from envs.open_spiel_env import OpenSpielEnv
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from games.registry import registry # Initilizes an empty registry dictionary
from games import loaders  # Adds the games to the registry dictionary
from utils.results_utils import print_total_scores



def initialize_environment(game, config: Dict[str, Any]) -> OpenSpielEnv:
    """Initializes the game environment."""
    player_types = [agent["type"] for agent in config["agents"]]
    return OpenSpielEnv(
        game=game,
        game_name=config["env_config"]["game_name"],
        player_types=player_types,
        max_game_rounds=config["env_config"].get("max_game_rounds"), # For iterrated games
    )


def create_agents(config: Dict[str, Any]) -> List:
    """Create agent instances based on configuration

    Args:
        config: Simulation configuration dictionary
        env: Initialized game environment

    Returns:
        List of agent instances

    Raises:
        ValueError: For invalid agent types or missing LLM models
    """
    # Instead of using a Dic, we use a list. This simplifies the naming and retrieval (?)
    agents = []

    for idx, agent_cfg in enumerate(config["agents"]):
        agent_type = agent_cfg["type"].lower()

        if agent_type == "human":
            agents.append(HumanAgent(game_name=config['env_config']['game_name']))
        elif agent_type == "random":
            agents.append(RandomAgent(seed=config.get("seed")))
        elif agent_type == "llm":
             agents.append(LLMAgent(
             llm = 'chatgpt',  #TODO: (lck) this should be a parameter in the config file
                game_name=config['env_config']['game_name']
            ))
        else:
            raise ValueError(f"Unsupported agent type: '{agent_type}'")

    return agents


def validate_config(config: Dict[str, Any]) -> None:
    """Validates the configuration."""
    game_name = config["env_config"]["game_name"]
    num_players = registry.get_game_loader(game_name)().num_players()

    if len(config["agents"]) != num_players:
        raise ValueError(
            f"Game '{game_name}' requires {num_players} players, "
            f"but {len(config['agents'])} agents were provided."
        )

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

    game_name = config["env_config"]["game_name"]

    # Set up logging
    logging.basicConfig(level=getattr(logging, config["log_level"].upper()))
    logger = logging.getLogger(__name__)
    logger.info("Starting simulation for game: %s", game_name)

    # 1. Set up random seed if specified
    if config.get("seed") is not None:
        random.seed(config["seed"])

    # TODO: FOR NOW THE GAME SIMULATOR IS NOT USED
    # 2. Load the pyspiel game object
    try:
        loader = registry.get_game_loader(game_name)
        game = loader()
    except ValueError as e:
        raise RuntimeError(f"Game loading failed: {str(e)}") from e

    # Initialize environment
    env = initialize_environment(game, config)

    # Agent setup
    agents = create_agents(config)

    # Run simulation loop
    all_episode_results, total_scores = simulate_episodes(env, agents, config)

    return {
        "game": game_name,
        "all_episode_results": all_episode_results,
        "total_scores": total_scores
    }

def simulate_episodes(env, agents, config):
    """
    Simulate multiple episodes.

    Args:
        env: The game environment.
        agents: A list of agents corresponding to players.
        episode: The current episode number.

    Returns:
        A dictionary containing the results of the episode.
    """

    # Initialize storage for episode results
    all_episode_results = []
    total_scores = {}  # To accumulate scores for all players

    for episode in range(config['num_episodes']):

        # Start a new episode
        observation = env.reset()  # board state and legal actions
        done =  env.state.is_terminal()

        # Play the game until it ends
        while not done:
            current_player = env.state.current_player()
            agent = agents[current_player]
            action = agent.compute_action(legal_actions=observation['legal_actions'],
                                        state= observation['state_string']
                        )

            # Step through the environment
            observation, rewards_dict, done, info = env.step(action)

        # Update results when the episode is finished
        all_episode_results.append({
            "episode": episode,
            "rewards": rewards_dict,
            "final_scores": list(info.values())[0]
        })

        for player, score in rewards_dict.items():
                    total_scores[player] = total_scores.get(player, 0) + score

    return all_episode_results, total_scores


def main():

    # Build the CLI parser
    parser = build_cli_parser()
    args = parser.parse_args()

    # Run the simulation
    result_dict = run_simulation(args)
    print_total_scores(result_dict['total_scores'])


if __name__ == "__main__":
    main()
