#!/usr/bin/env python3
"""
runner.py

Entry point for game simulations.
Handles Ray initialization, SLURM environment variables, and orchestration.
"""

import json
import argparse
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

    # TODO: add on SLURM  - Read game names from an environment variable (e.g., set via SLURM) or config.
    # TODO: add in the config a way to pass a list of games to simulate
    # TODO: add a debugger config!
    # TODO: each game should have its own: "max_game_rounds", "num_episodes", "agents" and "output_path"

    game_names = ["kuhn_poker"] # TODO: later delete this! - it should be a list of games!

    simulation_tasks = []
    for game in game_names:
        # (Optional) Adjust config["agents"] here for different matchups if needed.
        if use_ray:
            simulation_tasks.append(simulate_game_ray.remote(game, config, seed))
        else:
            simulation_tasks.append(simulate_game(game, config, seed))

    results = ray.get(simulation_tasks) if use_ray else simulation_tasks

    # TODO: implement SQL logging - and save the results in a SQL database
    output_path = config.get("output_path", "results/simulation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Simulation results saved to {output_path}")
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
        print("Simulation completed.")
    finally:
        full_cleanup()
