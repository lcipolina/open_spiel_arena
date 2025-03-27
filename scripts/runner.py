#!/usr/bin/env python3
"""
runner.py

Entry point for game simulations.
Handles Ray initialization, SLURM environment variables, and orchestration.
"""

# Suppress errors from DynamoRIO - TODO: delete this!!
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import sys
sys.path.insert(0, "/p/project1/ccstdl/cipolina-kun1/open_spiel_arena")

import os
import json
import argparse
import subprocess 
import logging
from typing import Dict, Any, List, Tuple
import ray
from configs.configs import parse_config
from utils.cleanup import full_cleanup
from utils.seeding import set_seed
from simulate import simulate_game

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def initialize_ray():
    """Initializes Ray if not already initialized."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized.")

@ray.remote
def simulate_game_ray(game_name: str, config: Dict[str, Any], seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ray remote wrapper for parallel game simulation.
    Calls the standard simulate_game function.
    """
    return simulate_game(game_name, config, seed)

def run_simulation(args):
    """
    Orchestrates simulation runs across multiple games and agent matchups.
    Reads the configuration, sets up Ray if enabled, and collects simulation results.
    """

    config = parse_config(args)
    seed = config.get("seed", 42)
    set_seed(seed)

    use_ray = config.get("use_ray", True)
    if use_ray:
        initialize_ray()

    # Extract list of games and their configurations
    game_configs = config.get("env_configs", [])

    simulation_tasks = []
    results = []
    for game_config in game_configs:
        game_name = game_config["game_name"]

        # Merge base config with game-specific config
        game_specific_config = {
            **config,  # Inherit global settings
            "env_config": game_config,  # Override only env_config for this game
            "max_game_rounds": game_config.get("max_game_rounds", None),
            "num_episodes": game_config.get("num_episodes", config.get("num_episodes", 1)),
            "agents": game_config.get("agents", config.get("agents", {})),
            "output_path": game_config.get("output_path", f"results/{game_name}_simulation_results.json"),
        }

        if use_ray:
            simulation_tasks.append(
                simulate_game_ray.remote(game_name, game_specific_config, seed)
            )
        else:
            results.append(simulate_game(game_name, game_specific_config, seed))

    if use_ray:
        results = ray.get(simulation_tasks)

    # Save results per game
    for game_result, game_config in zip(results, game_configs):
        output_path = game_config.get("output_path", f"results/{game_config['game_name']}_simulation_results.json")
        with open(output_path, "w") as f:
            json.dump(game_result, f, indent=4)
        logger.info(f"Simulation results for {game_config['game_name']} saved to {output_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM game simulations.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    parser.add_argument(
        "--override", nargs="*", metavar="KEY=VALUE",
        help="Key-value overrides for configuration (e.g., game_name=tic_tac_toe)."
    )
    args = parser.parse_args()
    try:
        run_simulation(args)
        print("Running post-game processing...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "..", "analysis", "post_game_processing.py")
        subprocess.run(["python", script_path], check=True)

        print("Simulation completed.")
    finally:
       full_cleanup()
