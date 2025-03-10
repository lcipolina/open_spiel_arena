import os
import sys
import json
import argparse
import ray
import logging
from typing import Dict, Any, List, Tuple
from agents import initialize_agents, setup_agents
from simulation import compute_actions
from utils import detect_illegal_moves
from envs.open_spiel_env import OpenSpielEnv
from games.registry import registry
from utils.cleanup import full_cleanup
from utils.seeding import set_seed

# Initialize Ray only if needed
def initialize_ray():
    """Initializes Ray if the configuration allows."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

@ray.remote
def simulate_game_ray(game_name: str, config: Dict[str, Any], seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    """Ray-based parallel execution of game simulation."""
    return simulate_game(game_name, config, seed)  # Calls the standard function

def simulate_game(game_name: str, config: Dict[str, Any], seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Runs a single game simulation with multiple episodes.
    """
    set_seed(seed)
    env = OpenSpielEnv(game_name, config, seed)
    agents = initialize_agents(config, seed)

    game_results = []
    for episode in range(config["num_episodes"]):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            actions = compute_actions(env, {i: agent for i, agent in enumerate(agents)}, observation)
            observation, _, done, _ = env.step(actions)
        game_results.append({"game": game_name, "players": {i: agent.get_metrics() for i, agent in enumerate(agents)}})

    return game_name, game_results

def run_simulation(args):
    """Runs game simulations for all LLM models and matchups."""
    config = json.load(open(args.config))
    seed = config.get("seed", 42)
    set_seed(seed)

    use_ray = config.get("use_ray", True)  # Toggle Ray ON/OFF

    if use_ray:
        initialize_ray()

    game_names = os.getenv("GAME_NAMES", "kuhn_poker,matrix_rps,tic_tac_toe,connect_four").split(",")
    llm_models = list(LLM_REGISTRY.keys())

    all_simulations = []
    for game in game_names:
        for llm in llm_models:
            for opponent in (["random"] + llm_models):
                config["agents"] = setup_agents(config, game)

                if use_ray:
                    all_simulations.append(simulate_game_ray.remote(game, config, seed))
                else:
                    # Run sequentially if Ray is disabled
                    all_simulations.append(simulate_game(game, config, seed))

    if use_ray:
        results = ray.get(all_simulations)
    else:
        results = all_simulations 

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM game simulations.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    args = parser.parse_args()

    try:
        run_simulation(args)
    finally:
        full_cleanup()