#!/usr/bin/env python3
"""
simulate.py

Core simulation logic for a single game simulation.
Handles environment creation, policy initialization, and the simulation loop.
"""
import time
import logging
from typing import Dict, Any, List, Tuple
from utils.seeding import set_seed
from games.registry import registry # Games registry
from agents.llm_registry import LLM_REGISTRY,initialize_llm_registry
initialize_llm_registry() #TODO: fix this, I don't like it!
from agents.policy_manager import initialize_policies, policy_mapping_fn
from utils.loggers import SQLiteLogger
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def log_llm_action(agent_id: int,agent_model: str,observation: Dict[str, Any],chosen_action: int,reasoning: str,flag: bool = False) -> None:
    """Logs the LLM agent's decision."""
    logger.info(f"Board state: \n{observation['state_string']}")
    logger.info(f"Legal actions: {observation['legal_actions']}")
    logger.info(f"Agent {agent_id} ({agent_model}) chose action: {chosen_action} with reasoning: {reasoning}")
    if flag == True:
       logger.error(f"Terminated due to illegal move: {chosen_action}.")




def simulate_game(game_name: str, config: Dict[str, Any], seed: int) -> str:
    """
    Runs a game simulation, logs agent actions and final rewards to TensorBoard.

    Args:
        game_name: The name of the game.
        config: Simulation configuration.
        seed: Random seed for reproducibility.

    Returns:
        str: Confirmation that the simulation is complete.
    """

     # Initialize loggers for all agents
    logger.info(f"Initializing environment for {game_name}.")

    policies_dict = initialize_policies(config, game_name, seed) # Assign LLMs to players in the game and loads the LLMs into GPU memory. TODO: see how we assign different models into different GPUs.

    # Initialize loggers and writers for all agents
    agent_loggers_dict = {
        policy_name: SQLiteLogger(
            agent_type=config["agents"][agent_id]["type"],
            model_name=config["agents"][agent_id].get("model", "N/A").replace("-", "_")
        )
        for agent_id, policy_name in enumerate(policies_dict.keys())
    }
    writer = SummaryWriter(log_dir=f"runs/{game_name}") # Tensorboard writer

    # Run the simulation loop
    env = registry.make_env(game_name, config) # Loads the pyspiel game and the env simulator

    for episode in range(config["num_episodes"]):
        episode_seed = seed + episode
        observation_dict, _ = env.reset(seed=episode_seed)
        terminated = truncated = False

        logger.info(f"Episode {episode + 1} started with seed {episode_seed}.")
        turn = 0

        while not (terminated or truncated):
            actions = {}
            for agent_id, observation in observation_dict.items():
                policy_key = policy_mapping_fn(agent_id) # Map agentID to policy key
                policy = policies_dict[policy_key]  # Policy class
                agent_logger = agent_loggers_dict[policy_key]
                agent_type = config["agents"][agent_id]["type"]  # "llm", "random", "human"
                agent_model = config["agents"][agent_id].get("model", "N/A")  # Model name (for LLMs)

                start_time = time.perf_counter()
                action_metadata =  policy(observation) #Calls `__call__()` -> `_process_action()` -> `log_move()`
                duration = time.perf_counter() - start_time

                if isinstance(action_metadata, int): # Non-LLM agents
                    chosen_action = action_metadata
                    reasoning = "N/A"  # No reasoning for non-LLM agents
                else: # LLM agents
                    chosen_action = action_metadata.get("action", -1)  # Default to -1 if missing
                    reasoning = str(action_metadata.get("reasoning", "N/A") or "N/A")

                actions[agent_id] = chosen_action

                # Check if the chosen action is legal
                if chosen_action is None or chosen_action not in observation["legal_actions"]:
                    if agent_type == "llm":
                       log_llm_action(agent_id, agent_model, observation, chosen_action, reasoning, flag = True)
                    agent_logger.log_illegal_move(game_name=game_name, episode=episode + 1,turn=turn,
                                                   agent_id=agent_id, illegal_action=chosen_action,
                                                   reason=reasoning, board_state=observation["state_string"])
                    truncated = True
                    break  # exit the for-loop over agents

                # Loggins
                opponents = ", ".join(
                    f"{config['agents'][a_id]['type']}_{config['agents'][a_id].get('model', 'N/A').replace('-', '_')}"
                    for a_id in config["agents"] if a_id != agent_id
                )

                agent_logger.log_move(
                    game_name=game_name,
                    episode=episode + 1,
                    turn=turn,
                    action=chosen_action,
                    reasoning=reasoning,
                    opponent= opponents,  # Get all opponents
                    generation_time=duration,
                    agent_type=agent_type,
                    agent_model=agent_model,
                    seed = episode_seed
                )

                if agent_type == "llm":
                   log_llm_action(agent_id, agent_model, observation, chosen_action, reasoning)

            # Step forward in the environment #TODO: check if this works for turn-based games (track the agent playing)
            if not truncated:
                observation_dict, rewards_dict, terminated, truncated, _ = env.step(actions)
                turn += 1

        # Logging
        game_status = "truncated" if truncated else "terminated"
        logger.info(f"Game status: {game_status} with rewards dict: {rewards_dict}")

        for agent_id, reward in rewards_dict.items():
            policy_key = policy_mapping_fn(agent_id)
            agent_logger = agent_loggers_dict[policy_key]
            agent_logger.log_game_result(
                    game_name=game_name,
                    episode=episode + 1,
                    status=game_status,
                    reward=reward
                )
            # Tensorboard logging
            agent_type = config["agents"][agent_id]["type"]
            agent_model = config["agents"][agent_id].get("model", "N/A").replace("-", "_")
            tensorboard_key = f"{agent_type}_{agent_model}"
            writer.add_scalar(f"Rewards/{tensorboard_key}", reward, episode + 1)

        logger.info(f"Simulation for game {game_name}, Episode {episode + 1} completed.")
    writer.close()
    return "Simulation Completed"

# start tensorboard from the terminal:
# tensorboard --logdir=runs

# In the browser:
# http://localhost:6006/
