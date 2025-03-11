#!/usr/bin/env python3
"""
simulate.py

Core simulation logic for a single game simulation.
Handles environment creation, policy initialization, and the simulation loop.
"""

import logging
from typing import Dict, Any, List, Tuple

from utils.seeding import set_seed
from envs.env_initializer import env_creator  # Environment factory function
from games.registry import registry # Games registry
#from games import loaders  # Adds the games to the registry dictionary
from agents.llm_registry import LLM_REGISTRY,initialize_llm_registry
initialize_llm_registry() #TODO: fix this, I don't like it!
from agents.policy_manager import initialize_policies, policy_mapping_fn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def simulate_game(game_name: str, config: Dict[str, Any], seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Runs a single game simulation for multiple episodes.

    Args:
        game_name: The name of the game.
        config: Simulation configuration.
        seed: Random seed for reproducibility.

    Returns:
        Tuple containing the game name and a list of simulation results.
    """
    set_seed(seed)
    logger.info(f"Initializing environment for {game_name} with seed {seed}.")
    env = registry.make_env(game_name, config)
    policies = initialize_policies(config, game_name, seed) # Assign LLMs to players in the game and loads the LLMs into GPU memory. TODO: see how we assign different models into different GPUs.

    game_results = []
    for episode in range(config["num_episodes"]):
        observation_dict, _ = env.reset(seed=seed + episode)
        terminated = truncated = False

        logger.info(f"Episode {episode + 1} started.")
        while not (terminated or truncated):
            actions = {}
            for agent_id, observation in observation_dict.items():
                policy_key = policy_mapping_fn(agent_id) # Map agentID to policy key
                policy = policies[policy_key]  # Map environment agent IDs to policy keys.
                actions[agent_id] = policy.compute_action(observation)
                logger.debug(f"Agent {agent_id} ({policy_key}) selected action {actions[agent_id]}.")
            observation_dict, rewards, terminated, truncated, _ = env.step(actions)
        logger.info(f"Episode {episode + 1} ended. Rewards: {rewards}")

        # TODO: improve this - save results in a more structured way, add name of the agent and probably save into SQL data
        game_results.append({"game": game_name, "episodes": episode + 1, "rewards": rewards})

    logger.info(f"Simulation for {game_name} completed.")
    return game_name, game_results
