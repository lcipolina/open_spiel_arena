#!/usr/bin/env python3
"""
simulate.py

Runs game simulations with configurable agents and tracks outcomes.
Supports both CLI arguments and config dictionaries.
Includes performance reporting and logging.
"""

import logging
import random
from typing import Dict, Any, List, Union
from agents.agent_registry import AGENT_REGISTRY
from agents.agent_report import AgentPerformanceReporter

from configs.configs import build_cli_parser, parse_config, validate_config
from envs.open_spiel_env import OpenSpielEnv
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from agents.llm_utils import load_llm_from_registry
from games.registry import registry # Initilizes an empty registry dictionary
from games import loaders  # Adds the games to the registry dictionary
from utils.results_utils import print_total_scores


def initialize_environment(config: Dict[str, Any]) -> OpenSpielEnv:
    """Loads the game from pyspiel and initializes the game environment simulator."""

    # Load the pyspiel game object
    player_types = [agent["type"] for _, agent in sorted(config["agents"].items())]
    game_name = config["env_config"]["game_name"]
    game_loader = registry.get_game_loader(game_name)()

    # Load the environment simulator instance
    return registry.get_simulator_instance(
        game_name=game_name,
        game=game_loader,
        player_types= player_types,
        max_game_rounds=config["env_config"].get("max_game_rounds") # For iterated games
    )

# TODO: add done? terminated? to the base env class

## QUESTION: I am not sure if this should go here or in the 'agents' folder and imported, for modularity, vs readability?
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

    agents = []
    game_name = config["env_config"]["game_name"]

    # Iterate over agents in numerical order
    for _, agent_cfg in sorted(config["agents"].items()):
        agent_type = agent_cfg["type"].lower()

        if agent_type not in AGENT_REGISTRY:
                    raise ValueError(f"Unsupported agent type: '{agent_type}'")

        # Dynamically instantiate the agent class
        agent_class = AGENT_REGISTRY[agent_type]

        if agent_type == "llm":
                model_name = agent_cfg.get("model", "gpt2")
                llm = load_llm_from_registry(model_name)
                agents.append(agent_class(llm=llm, game_name=game_name))
        else:
                agents.append(agent_class(game_name=game_name))  # Other agent types


    return agents


# HERE I HAVE A PROBLEM BECAUSE IT NEEDS TO REDIRECT TO THE LLM Agent but with the specific prompt for the game!!
def _get_action(
    env: OpenSpielEnv, agents_list: List[Any], observation: Dict[str, Any]
) -> Union[List[int], int]:
    """
    Computes actions for all players involved in the current step.

    Args:
        env (OpenSpielEnv): The game environment.
        agents_list (List[Any]): List of agents corresponding to the players.
        observation (Dict[str, Any]): The current observation, including legal actions.

    Returns:
        int: The action selected by the current player (turn-based gamees).
        List[int]: The actions selected by the players (simultaneous move games).
    """

    # Handle sequential move games
    current_player = env.state.current_player()

    # Handle simultaneous move games
    if env.state.is_simultaneous_node():
        return [
            agent.compute_action(observation)
            for player, agent in enumerate(agents_list)
        ]
    elif current_player >= 0:  # Default players (turn-based)
        agent = agents_list[current_player]
        return agent.compute_action(observation)

def simulate_episodes(
    env: OpenSpielEnv, agents: List[Any], config: Dict[str, Any]
) -> Dict[str, Any]:
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

        # TODO: add 'terminated' as well! (for iterated games)

        # Play the game until it ends
        while not done:
            action = _get_action(env, agents, observation)
            observation, rewards_dict, done, info = env.step(action)

        # Update results when the episode is finished
        all_episode_results.append({
            "episode": episode,
            "rewards": rewards_dict,
        })

        for player, score in rewards_dict.items():
                    total_scores[player] = total_scores.get(player, 0) + score

    return all_episode_results, total_scores

def run_simulation(args) -> Dict[str, Any]:
    """
    Orchestrates the simulation workflow and generates reports.

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

    # Set up random seed
    if config.get("seed") is not None:
        random.seed(config["seed"])

    # Initialize environment
    env = initialize_environment(config)

    # Agent setup
    agents = create_agents(config)

    # Run simulation loop
    all_episode_results, total_scores = simulate_episodes(env, agents, config)

    # Print final board for the finished game
    print(f"Final game state:\n {env.state}")

    # Performance reports
    reporter = AgentPerformanceReporter(agents)
    reporter.collect_metrics()
    reporter.print_summary()
    reporter.plot_metrics()

    return {
        "game_name": game_name,
        "all_episode_results": all_episode_results,
        "total_scores": total_scores
    }


def main():

    # Build the CLI parser
    parser = build_cli_parser()
    args = parser.parse_args()

    # Run the simulation
    result_dict = run_simulation(args)
    print_total_scores(result_dict["game_name"],result_dict['total_scores'])

    # TODO: Save results in results/JSON file! together with the other things requested.


if __name__ == "__main__":
    main()
