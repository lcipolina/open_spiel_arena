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
from games.registry import registry # Initilizes an empty registry dictionary for the games
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

def initialize_agents(config: Dict[str, Any]) -> List:
    """Create agent instances based on configuration

    Args:
        config: Simulation configuration dictionary

    Returns:
        List of agent instances

    Raises:
        ValueError: For invalid agent types or missing LLM models
    """
    agents_list = []
    game_name = config["env_config"]["game_name"]

    # Iterate over agents in numerical order
    for _, agent_cfg in sorted(config["agents"].items()):
        agent_type = agent_cfg["type"].lower()

        if agent_type not in AGENT_REGISTRY:
            raise ValueError(f"Unsupported agent type: '{agent_type}'")

        # Dynamically instantiate the agent class
        agent_class = AGENT_REGISTRY[agent_type]

        if agent_type in ["llm", "human"]:
            model_name = agent_cfg.get("model", "gpt2")
            agents_list.append(agent_class(model_name=model_name, game_name=game_name))
        elif agent_type == "random":
             seed = config.get("seed")
             agents_list.append(agent_class(seed=seed))
        else:
            try:
              agents_list.append(agent_class(game_name=game_name))
            except TypeError:
              agents_list.append(agent_class())
    return agents_list

def _get_action(
    env: OpenSpielEnv, player_to_agent: Dict[int, Any], observation: Dict[str, Any]
) -> Dict[int, int]:
    """
    Computes actions for all (shuffled) players involved in the current step.

    Args:
        env (OpenSpielEnv): The game environment.
        player_to_agent (Dict[int, Any]): Mapping from OpenSpiel player index to shuffled agent.
        observation (Dict[str, Any]): 'state_string' and 'legal_actions'.

    Returns:
        Dict[int, int]: A dictionary mapping player indices to selected actions.
    """

    # Handle simultaneous-move games (all players act at once)
    if env.state.is_simultaneous_node():
        return {
            player: player_to_agent[player](observation[player]) # CALL agent function to get action
            for player in player_to_agent
            }

    # Handle turn-based games (only one player acts)
    current_player = env.state.current_player()
    return {current_player: player_to_agent[current_player](observation[current_player])}

def simulate_episodes(
    env: OpenSpielEnv, agents: List[Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulate multiple episodes.

    Args:
        env: The game environment.
        agents: A list of agents classes corresponding to players.
        config: Simulation configuration.

    Returns:
        A dictionary containing the results of the episode.
    """

    # Initialize storage for episode results
    all_episode_results = []
    total_scores = {agent: 0 for agent in agents}  # Track scores per agent

    for episode in range(config['num_episodes']):

        # Shuffle agent and map from OpenSpiel indices to shuffled agents
        shuffled_agents = random.sample(agents, len(agents))
        player_to_agent = {player_idx: shuffled_agents[player_idx] for player_idx in range(len(shuffled_agents))}
        actions_dict = {idx: None for idx in player_to_agent}

        # Start a new episode
        observation_dict, info = env.reset()  # board state and legal actions

        terminated = False  # Whether the episode has ended normally
        truncated = False  # Whether the episode ended due to `max_game_rounds`

        # Play the game until it ends
        while not (terminated or truncated):
            actions_dict = _get_action(env, player_to_agent, observation_dict)
            observation_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict) # Actions passed as {player: action}

        # Update results when the episode is finished
        for player, reward in rewards_dict.items():
            total_scores[player_to_agent[player]] += reward   # Map rewards back to agents

        all_episode_results.append({
            "episode": episode,
            "rewards": {player_to_agent[player]: reward for player, reward in rewards_dict.items()},
            "terminated": terminated,
            "truncated": truncated,
        })

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

    # Initialize agents list
    agents_list = initialize_agents(config)

    # Run simulation loop
    all_episode_results, total_scores = simulate_episodes(env, agents_list, config)

    # Print final board for the finished game
    print(f"Final game state:\n {env.state}")

    # Performance reports
    reporter = AgentPerformanceReporter(agents_list)
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

    # TODO: Save results in results/JSON file with payer's names! together with the other things requested.


if __name__ == "__main__":
    main()
