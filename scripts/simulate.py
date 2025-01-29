#!/usr/bin/env python3
"""
simulate.py

Runs game simulations with configurable agents and tracks outcomes.
Supports both CLI arguments and config dictionaries.
"""

import logging
import random
from typing import Dict, Any, List
from enum import Enum, unique

from configs.configs import build_cli_parser, parse_config, validate_config
from envs.open_spiel_env import OpenSpielEnv
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from agents.llm_utils import load_llm_from_registry
from games.registry import registry # Initilizes an empty registry dictionary
from games import loaders  # Adds the games to the registry dictionary
from utils.results_utils import print_total_scores

'''
@unique
class PlayerId(Enum):
    CHANCE = -1
    SIMULTANEOUS = -2
    INVALID = -3
'''

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

    # Iterate over agents in numerical order
    for _, agent_cfg in sorted(config["agents"].items()):
        agent_type = agent_cfg["type"].lower()

        if agent_type == "human":
            agents.append(HumanAgent(game_name=config['env_config']['game_name']))
        elif agent_type == "random":
            agents.append(RandomAgent(seed=config.get("seed")))
        elif agent_type == "llm":
                    model_name = agent_cfg.get("model", "gpt2")  # Default to "gpt2" if no model is specified
                    llm = load_llm_from_registry(model_name)
                    agents.append(LLMAgent(llm=llm, game_name=config['env_config']['game_name']))
        # elif agent_type == "trained":
        #     agents_dict[p_name] = TrainedAgent("checkpoint.path")
        else:
            raise ValueError(f"Unsupported agent type: '{agent_type}'")

    return agents

#TODO: this is needed because of OpenSpiels ambiguous representation of the playerID - to check if we can delete
'''
def normalize_player_id(self,player_id):
    """Normalize player_id to its integer value for consistent comparisons.

    This is needed as OpenSpiel has ambiguous representation of the playerID

    Args:
        player_id (Union[int, PlayerId]): The player ID, which can be an
                integer or a PlayerId enum instance.
    Returns:
            int: The integer value of the player ID.
        """
    if isinstance(player_id, PlayerId):
        return player_id.value  # Extract the integer value from the enum
    return player_id  # If already an integer, return it as is
'''

def _get_action(
    env: OpenSpielEnv, agents_list: List[Any], observation: Dict[str, Any]
) -> List[int]:
    """
    Computes actions for all players involved in the current step.

    Args:
        env (OpenSpielEnv): The game environment.
        agents_list (List[Any]): List of agents corresponding to the players.
        observation (Dict[str, Any]): The current observation, including legal actions.

    Returns:
        List[int]: The action(s) selected by the players.
    """

    # Handle sequential move games
    current_player = env.state.current_player()
   # player_id = normalize_player_id(current_player)

    # Handle simultaneous move games
    if env.state.is_simultaneous_node():
        return [
            agent.compute_action(
                legal_actions=observation["legal_actions"][player],
                state=observation.get("state_string"),
                info = observation.get("info",None)
            )
            for player, agent in enumerate(agents_list)
        ]

    # Handle chance nodes where the environment acts randomly.
    elif env.state.is_chance_node():
        outcomes, probabilities = zip(*env.state.chance_outcomes())
        action = random.choices(outcomes, probabilities, k=1)[0]
        return [action]


    elif current_player >= 0:  # Default players (turn-based)
        agent = agents_list[current_player]
        return [
            agent.compute_action(
                legal_actions=observation["legal_actions"],
                state=observation.get("state_string"),
                info = observation.get("info",None)
            )
        ]

'''
# Collect actions
                current_player = state.current_player()
                player_id = self.normalize_player_id(current_player)

                if player_id == PlayerId.CHANCE.value:
                    # Handle chance nodes where the environment acts randomly.
                    self._handle_chance_node(state)
                elif player_id == PlayerId.SIMULTANEOUS.value:
                     # Handle simultaneous moves for all players.
                    actions = self._collect_actions(state)
                    state.apply_actions(actions)
                elif player_id == PlayerId.TERMINAL.value:
                    break
                elif current_player >= 0:  # Default players (turn-based)
                    legal_actions = state.legal_actions(current_player)
                    action = self._get_action(current_player, state, legal_actions)
                    state.apply_action(action)
'''

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

    return {
        "game_name": game_name,
        "all_episode_results": all_episode_results,
        "total_scores": total_scores
    }

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
