#!/usr/bin/env python3
# utils/game_utils.py
"""
simulate.py

Runs game simulations with configurable agents and tracks outcomes.
Supports both CLI arguments and config dictionaries.
Includes performance reporting and logging.
"""

import logging
import random
from typing import Dict, Any, List, Tuple
from agents.agent_registry import AGENT_REGISTRY
# from agents.agent_report import AgentPerformanceReporter  #TODO: delete this!
from configs.configs import build_cli_parser, parse_config, validate_config
from envs.open_spiel_env import OpenSpielEnv
from games.registry import registry # Initilizes an empty registry dictionary for the games
from games import loaders  # Adds the games to the registry dictionary
from utils.results_utils import print_total_scores
from utils.loggers import log_simulation_results, time_execution


def detect_illegal_moves(env: OpenSpielEnv, actions_dict: Dict[int, int]) -> int:
    """
    Detects illegal moves by comparing chosen actions with OpenSpiel's legal actions.

    Args:
        env: The game environment.
        actions_dict: Dictionary mapping player IDs to chosen actions.

    Returns:
        int: The number of illegal moves detected.
    """
    illegal_moves = 0
    for player, action in actions_dict.items():
        legal_actions = env.state.legal_actions(player)
        if action not in legal_actions:
            illegal_moves += 1
    return illegal_moves


def get_episode_results(rewards_dict: Dict[int, float], episode_players: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes episode results for all players.

    Args:
        rewards_dict: Dictionary mapping player IDs to their rewards.
        episode_players: Dictionary mapping player IDs to their type and model.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing player results.
    """
    return [
        {
            "player_id": player_idx,
            "player_type": player_data["player_type"],
            "player_model": player_data["player_model"],
            "result": "win" if rewards_dict.get(player_idx, 0) > 0 else
                      "loss" if rewards_dict.get(player_idx, 0) < 0 else "draw"
        }
        for player_idx, player_data in episode_players.items()
    ]


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
    # We rely on OpenSpiel's internal state to determine the current player
    # then we map to our shuffled agents
    current_player = env.state.current_player()
    return {current_player: player_to_agent[current_player](observation[current_player])}

def simulate_episodes(
    env: OpenSpielEnv, agents: List[Any], config: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Simulates multiple episodes and logs game events.

    Args:
        env: The game environment.
        agents: A list of agents classes corresponding to players.
        config: Simulation configuration.

    Returns:
         Tuple[str, List[Dict[str, Any]]]: Model name(s) and game results.
    """

    # Initialize storage for episode results
    game_results = []
    game_name = config["env_config"]["game_name"]
    agent_configs = config["agents"]

    for _ in range(config['num_episodes']):

        # Shuffle agent and map from OpenSpiel indices to shuffled agents
        # the _get_action function will use this mapping to get actions from OS internal idx to shuffled agents.
        shuffled_agents = random.sample(agents, len(agents))
        player_to_agent = {player_idx: shuffled_agents[player_idx] for player_idx in range(len(shuffled_agents))}
        actions_dict = {idx: None for idx in player_to_agent}

        # Track each player's type and model for this episode
        episode_players = {
            player_id: {
                "player_type": agent_configs[player_id]["type"],
                "player_model": agent_configs[player_id].get("model", "None")
            }
            for player_id in player_to_agent.keys()
        }

        # Start a new episode
        observation_dict, _ = env.reset()  # board state and legal actions
        moves = []
        illegal_moves, rounds = 0, 0
        terminated = False  # Whether the episode has ended normally
        truncated = False  # Whether the episode ended due to `max_game_rounds`

        # Play the game until it ends
        # observation_dict brings the new state and legal actions for all players
        # then _get_action maps to the corresponding shuffled agents.
        while not (terminated or truncated):
            actions_dict = _get_action(env, player_to_agent, observation_dict) # actions_dict =  {player_id: action}

            # Detect illegal moves
            illegal_moves += detect_illegal_moves(env, actions_dict)

            moves.append(actions_dict)
            rounds += 1
            observation_dict, \
            rewards_dict, \
            terminated, \
            truncated, \
            _ = env.step(actions_dict)   # observations_dict =  {player_id: observation}


        # Update results when the episode is finished - TODO: check this against all games
        episode_results = get_episode_results(rewards_dict, episode_players)

        game_results.append({
            "game": game_name,
            "rounds": rounds,
            "moves": moves,
            "illegal_moves": illegal_moves,
            "players": episode_results
        })

        # Identify all LLM models used
        llm_models = [
            data.get("model", "None")
            for data in agent_configs.values()
            if data["type"] == "llm"
        ]
        model_name = ", ".join(llm_models) if llm_models else "None"

        return model_name, game_results



@time_execution
@log_simulation_results
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

    logging.basicConfig(level=getattr(logging, config["log_level"].upper()))
    logger = logging.getLogger(__name__)
    logger.info("Starting simulation for game: %s", game_name)

    if config.get("seed") is not None:
        random.seed(config["seed"])

    env = initialize_environment(config)
    agents_list = initialize_agents(config)

    # Run simulation loop
    model_name, game_results = simulate_episodes(env, agents_list, config)

    # Print final game state
    print(f"\nFinal game state:\n{env.state}")

    # Print game outcomes
    for result in game_results:
        print(f"\nGame: {result['game']}")
        print(f"Rounds played: {result['rounds']}")
        print(f"Illegal moves: {result['illegal_moves']}")

        # Instead of accessing a non-existent 'result' key, loop through players
        for player in result["players"]:
            print(f"Player {player['player_id']} ({player['player_type']} - {player['player_model']}): {player['result'].upper()}")

    return model_name, game_results


def main():
    """Main entry point for the simulation script."""

    # Build the CLI parser
    parser = build_cli_parser()
    args = parser.parse_args()

    # Run the simulation
    run_simulation(args)


if __name__ == "__main__":
    main()
