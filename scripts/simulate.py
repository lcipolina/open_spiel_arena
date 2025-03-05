#!/usr/bin/env python3

"""
simulate.py

Runs game simulations with configurable agents and tracks outcomes.
Supports both CLI arguments and config dictionaries.
Includes performance reporting and logging.
"""



print("Running simulate.py...")

import os, sys

# This is just to load the environment   - not sure if it is needed.
import subprocess

# Define the paths
mamba_path = "/p/scratch/laionize/cache-kun1/miniconda3/bin/activate"
env_path = "/p/scratch/laionize/cache-kun1/llm"

# Command to activate Mamba and run Python inside the environment
command = f"source {mamba_path} {env_path} && python -c 'import sys; print(sys.executable)'"

# Run the command inside a Bash shell
result = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True)

print("Output:", result.stdout.strip())



# Dynamically add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ray
import argparse
import logging  # this is not used, TODO: I believe we need to initialize the logging?
import random
import json
from typing import Dict, Any, List, Tuple
from agents.agent_registry import AGENT_REGISTRY
from configs.configs import parse_config, validate_config  #TODO: delete this module or call this validate later
from envs.open_spiel_env import OpenSpielEnv
from games.registry import registry # Initilizes an empty registry dictionary for the games
from agents.llm_registry import LLM_REGISTRY,cleanup_vllm,initialize_llm_registry, close_simulation
from games import loaders  # Adds the games to the registry dictionary
# from utils.loggers import log_simulation_results, time_execution #TODO: delete this!
from utils.seeding import set_seed

initialize_llm_registry() #TODO: fix this, I don't like it!

# Load SLURM Output Path from Environment
OUTPUT_PATH = os.getenv(
    "OUTPUT_PATH",
    "/p/project/ccstdl/cipolina-kun1/open_spiel_arena/results/simulation_results.json"
)


'''
# Initialize Ray from SLURM either local or distributed mode.
#if ray.is_initialized(): # commented for faster debugging
#   ray.shutdown()
if os.getenv("DEBUG", "0") == "1":
    print("Debug mode enabled: Running Ray in local mode (single process)")
    #ray.init(local_mode=True,runtime_env={"env_vars": {"PYTHONPATH": "/p/project/ccstdl/cipolina-kun1/open_spiel_arena"}})
    # Force Ray workers to use the same Python executable
    os.environ["PYTHON_EXECUTABLE"] = sys.executable

    ray.init(
        local_mode=True,
        runtime_env={
            "env_vars": {
                "PYTHON_EXECUTABLE": sys.executable,  # Ensure all Ray workers use this Python
                "PYTHONPATH": "/p/project/ccstdl/cipolina-kun1/open_spiel_arena"
            }
        }
    )

else:
    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                "PYTHON_EXECUTABLE": sys.executable,  # Ensure all Ray workers use this Python
                "PYTHONPATH": "/p/project/ccstdl/cipolina-kun1/open_spiel_arena"
            }
        }
    )
'''
def detect_illegal_moves(env: OpenSpielEnv, actions_dict: Dict[int, int]) -> int:
    """
    Detects illegal moves by comparing chosen actions with OpenSpiel's legal actions.

    Args:
        env: The game environment.
        actions_dict: Dictionary mapping player IDs to chosen actions.

    Returns:
        int: The number of illegal moves detected.
    """
    return sum(
        1 for player, action in actions_dict.items()
        if action not in env.state.legal_actions(player)
    )

#TODO: use this in the code!!
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

def initialize_environment(config: Dict[str, Any], seed:int) -> OpenSpielEnv:
    """Loads the game from pyspiel and initializes the game environment simulator."""

    # Load the pyspiel game object
    player_types = [agent["type"] for _, agent in sorted(config["agents"].items())]
    game_name = config["env_config"]["game_name"]
    game_loader = registry.get_game_loader(game_name)()

    # Load the environment simulator instance
    env =  registry.get_simulator_instance(
        game_name=game_name,
        game=game_loader,
        player_types= player_types,
        max_game_rounds=config["env_config"].get("max_game_rounds"), # For iterated games
        seed=seed

    )

    return env

def initialize_agents(config: Dict[str, Any], seed:int) -> List:
    """
    Initializes the agents classes (i.e policies) based on the configuration.

    Args:
        config (Dict[str, Any]): Simulation configuration.
        game_name (str): The game being played.

    Returns:
        List: A list of agent class instances.

    Raises:
        ValueError: If an invalid agent type or missing model is found.
    """
    agents_list = []
    game_name = config["env_config"]["game_name"]

    for agent in config["agents"].values():
        agent_type = agent["type"].lower()

        if agent_type not in AGENT_REGISTRY:
            raise ValueError(f"Unsupported agent type: '{agent_type}'")

        agent_class = AGENT_REGISTRY[agent_type]  # Loads agent's base class

        if agent_type == "llm":
            model_name = agent.get("model")
            agents_list.append(agent_class(model_name=model_name, game_name=game_name))
        elif agent_type == "random":
            agents_list.append(agent_class(seed=seed))
        else:
            agents_list.append(agent_class())

    return agents_list # list of base classes for each of the agent's type on the config dict

def setup_agents(config: Dict[str, Any], game_name: str) -> Dict[int, Dict[str, str]]:
    """
    Assigns agents (llm, random, human) to players and updates the config dict with the assigned agents.
    - Uses manually assigned agents from config if present.
    - Dynamically assigns agents if missing.
    - Overrides the config dict with the updated agents.
    - Ensures correct number of agents per game.

    Args:
        config (Dict[str, Any]): Full simulation configuration.
        game_name (str): The game being played.

    Returns:
        Dict[int, Dict[str, str]]: Agent model per player.
    """
    num_players = registry.get_game_loader(game_name)().num_players()
    mode = config.get("mode", "llm_vs_random")

    # Retrieve all registered models
    llm_models = list(LLM_REGISTRY.keys())

    # If manually set in config, use it
    if mode == "manual":
        if "agents" not in config or len(config["agents"]) != num_players:
            raise ValueError(
                f"Manual mode requires explicit agent definitions. "
                f"Expected {num_players} agents, but got {len(config.get('agents', {}))}."
            )
        return config["agents"]

    # Otherwise, dynamically assign agents
    agents = {}
    if mode == "llm_vs_random":
        for i in range(num_players):
            agents[i] = {
                "type": "llm" if i == 0 else "random",
                "model": llm_models[i % len(llm_models)] if i == 0 else "None"
            }
    elif mode == "llm_vs_llm":
        for i in range(num_players):
            agents[i] = {
                "type": "llm",
                "model": llm_models[i % len(llm_models)]
            }

    # Apply overrides if provided in the config
    if "agents" in config:
        for key, value in config["agents"].items():
            agents[int(key)] = value

    # Persist changes
    config["agents"] = agents

    return config["agents"]


def compute_actions(
    env: OpenSpielEnv, player_to_agent: Dict[int, Any], observations: Dict[str, Any]
) -> Dict[int, int]:
    """
    Computes actions for all players using batch processing where applicable.
    Each agent handles its own decision logic.

    Args:
        env (OpenSpielEnv): The game environment.
        player_to_agent (Dict[int, Any]): Mapping from player index to agent.
        observations (Dict[str, Any]): Dictionary with state and legal actions.

    Returns:
        Dict[int, int]: A dictionary mapping player indices to selected actions.
    """

    # Simultaneous-move game: All players act at once  #TODO: test this!!
    if env.state.is_simultaneous_node():
        return {player: player_to_agent[player](observations[str(player)]) for player in player_to_agent}

    # Turn-based game: Only the current player acts
    current_player = env.state.current_player()
    return {current_player: player_to_agent[current_player](observations[current_player])}

def compute_actions_old(
    env: OpenSpielEnv, player_to_agent: Dict[int, Any], observations: Dict[str, Any]
) -> Dict[int, int]:
    """
    Computes actions for all players using batch processing where applicable.
    Each agent handles its own prompt logic.

    Args:
        env (OpenSpielEnv): The game environment.
        player_to_agent (Dict[int, Any]): Mapping from OpenSpiel player index to shuffled agent.
        observations (Dict[str, Any]): 'state_string' and 'legal_actions' for each player.

    Returns:
        Dict[int, int]: A dictionary mapping player indices to selected actions.
    """

    # Simultaneous-move game: All players act at once
    if env.state.is_simultaneous_node():
        actions, prompts, legal_actions, model_names = {}, {}, {}, {}

        # Collect all LLM requests first
        for player, agent in player_to_agent.items():
            obs = observations[str(player)]
            if agent.agent_type == "llm":
                prompts[player] = agent(obs)
                legal_actions[player] = tuple(obs["legal_actions"])
                model_names[player] = agent.model_name
            else:
                actions[player] = agent(obs)  # Non-LLM agents act immediately

        # Batch process LLM actions if applicable
        if prompts:
            llm_moves = ray.get(batch_llm_decide_moves.remote(model_names, prompts, legal_actions))
            actions.update(llm_moves)

        return actions

    # Turn-based game: Only the current player acts
    current_player = env.state.current_player()
    return {current_player: player_to_agent[current_player](observations[current_player])}


# @ray.remote # Runs on its own ray worker #TODO: just for debugging
def simulate_game(game_name: str,
                  config: Dict[str, Any],
                  seed: int
                  ) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Runs multiple episodes of a single game in parallel.

    Args:
        game_name (str): The game to simulate.
        config (Dict[str, Any]): Simulation configuration.
        seed: Random seed.

    Returns:
        Tuple[str, List[Dict[str, Any]]]: Model name(s) and game results.
    """

    set_seed(seed)

    config["agents"] = setup_agents(config, game_name) # Assign agents (models) to players
    env = initialize_environment(config, seed) # Loads game and simulation environment
    agents = initialize_agents(config, seed)  # list of base classes (policies) for each agent type on the config dict

    game_results = []
    try:
        for episode in range(config["num_episodes"]):
            observation_dict, _ = env.reset(seed=seed + episode) # Each episode has a different seed
            actions = {}
            terminated = False  # Whether the episode has ended normally
            truncated = False  # Whether the episode ended due to `max_game_rounds`

            # Map players to agents
            #shuffled_agents = random.sample(agents, len(agents))
            #player_to_agent = {player_idx: shuffled_agents[player_idx] for player_idx in range(len(shuffled_agents))}

            player_to_agent = {player_idx: agents[player_idx] for player_idx in range(len(agents))}
            while not (terminated or truncated):
                actions = compute_actions(env, player_to_agent, observation_dict) # Get batched actions for all players
            # illegal_moves = detect_illegal_moves(env, actions)  # Detect illegal moves  #TODO: see this!
            # if illegal_moves:
            #     logging.warning("Illegal moves detected: %d", illegal_moves)
                observation_dict, rewards, terminated, truncated, _ = env.step(actions)
                if terminated or truncated:
                    break

            game_results.append({
                "game": game_name,
                "rounds": len(actions),
                "players": {idx: agent.get_performance_metrics() for idx, agent in enumerate(agents)}
            })

    finally:
        # Decide whether to clean up the model based on simulation mode
        if config["mode"] == "llm_vs_llm":
            print("Keeping LLM in memory for next game...")
        else:
           # cleanup_vllm(CURRENT_LLM) #TODO: see this, i am not sure
            CURRENT_LLM = None #TODO: I am not sure this is used

    # Identify all LLM models used
    llm_models_used = [
        data.get("model", "None") for data in config["agents"].values() if data["type"] == "llm"
    ]
    model_name = ", ".join(llm_models_used) if llm_models_used else "None"

    return model_name, game_results #TODO: see what happens with the rewards


#################################
######### MAIN FUNCTION #########
#################################

def run_simulation(args):
    """Main function to run the simulation across
    1. Each game
    2. Each LLM model
    3. LLM vs Random bot
    4. LLM vs Every other LLM
    """

    config = parse_config(args)

    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)  # Defaults to INFO if invalid
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)  #TODO: see what to do with this


    seed = config.get("seed", 42)
    set_seed(seed)

    # Read game names from SLURM environment variable (if set)
    game_names = os.getenv("GAME_NAMES", "kuhn_poker,matrix_rps,tic_tac_toe,connect_four").split(",")

    try:
        # Run simulations in parallel (Ray)
        results = simulate_game("kuhn_poker", config, seed)      # Without Ray for debugging: TODO: delete this!
        #results = ray.get([simulate_game.remote(game, config, seed) for game in game_names])
    finally:
        # At the end of ALL simulations, free GPU memory
       # close_simulation() #TODO: delete this once finish debugging
       a = 0


    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM game simulations.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    parser.add_argument(
        "--override", nargs="*", metavar="KEY=VALUE",
        help="Key-value overrides for configuration (e.g., game_name=tic_tac_toe)."
    )
    args = parser.parse_args()


    run_simulation(args)

'''

oki doki, now that is working! Let's proceed on thinking our loop. We need to run the following:

1) For every game
2) For every LLM model
3) Run against a random agent
4) Run against every other LLM

I believe we only have the game loop, but not the loop (3) and (4)

I cna re-share the 'simulate.py' script if needed

'''