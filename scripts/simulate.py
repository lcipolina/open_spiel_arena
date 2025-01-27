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
from utils.common_utils import print_simulation_summary, validate_config



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


def create_agents_old(config: Dict[str, Any], env: OpenSpielEnv) -> Dict[str, Any]:
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
                game_name=config['env_config']['game_name']
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
                llm = 'chatgpt',
                game_name=config['env_config']['game_name']
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
    # validate_config(config) #TODO: (lck) see if this needs to be removed or check for other config entries

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
    agents = create_agents(config)

    # Run simulation loop
    results = simulate_episodes(env, agents, config)

    return {
        "game": config.game_name,
        "results": results
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
    for episode in range(config['num_episodes']):

        # Start a new episode
        observation = env.reset()  # board state and legal actions
        done = False
        episode_result = {"episode": episode}  # Initialize with episode number

        # Play the game until it ends
        while not done:
            current_player = env.state.current_player()
            agent = agents[current_player]

            # Agent decides the action
            action = agent.compute_action(legal_actions=observation['legal_actions'],
                                        state= observation['state_string']
                        )

            # Step through the environment
            observation, reward_dict, done, info = env.step(action)

            # Update results when the episode is finished
            if done:
                episode_result.update({
                    "winner": info.get("winner"),
                    "scores": info.get("scores")
                })
    return episode_result


def main():

    # Build the CLI parser
    parser = build_cli_parser()
    args = parser.parse_args()

    # Run the simulation
    results = run_simulation(args)
    print_simulation_summary(results)


if __name__ == "__main__":
    main()
