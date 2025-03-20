'''
policy_manager.py

Policy assignment, initialization, and mapping.
'''


import logging
from typing import Dict, Any
from agents.agent_registry import AGENT_REGISTRY
from agents.llm_registry import LLM_REGISTRY
from games.registry import registry

# Configure logger
logger = logging.getLogger(__name__)

def initialize_policies(config: Dict[str, Any], game_name: str, seed: int) -> Dict[str, Any]:
    """
    Dynamically assigns policies to agents and initializes them in a single step.

    This function:
      - Ensures `config["agents"]` is correctly assigned.
      - Initializes policies for all assigned agents.
      - Returns a dictionary of initialized policies keyed by policy IDs.

    Args:
        config (Dict[str, Any]): Simulation configuration.
        game_name (str): The game being played.
        seed (int): Random seed.

    Returns:
        Dict[str, Any]: A dictionary mapping policy names to agent instances.
    """
    # Assign LLM models to players in the game
    num_players = registry.get_game_loader(game_name)().num_players()
    mode = config.get("mode", "llm_vs_random")
    llm_models = list(LLM_REGISTRY.keys())

    if "agents" not in config or len(config["agents"]) != num_players:
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
        # Apply manual overrides if present
        if "agents" in config:
            for key, value in config["agents"].items():
                agents[int(key)] = value

        config["agents"] = agents  # Update config with agent assignments #TODO: see if we still need to update the config dict or we can remove this logic
        logger.info(f"Agent setup completed for mode: {mode}. Assigned agents: {agents}")

    # Initialize policies based on assigned agents - Loads the LLMs into GPU memory
    policies = {}
    for i in range(num_players):
        agent_config = config["agents"].get(i, {"type": "random"})
        agent_type = agent_config["type"].lower()

        if agent_type not in AGENT_REGISTRY:
            raise ValueError(f"Unsupported agent type: '{agent_type}'")

        agent_class = AGENT_REGISTRY[agent_type]

        if agent_type == "llm":
            model_name = agent_config.get("model", list(LLM_REGISTRY.keys())[0])
            policies[f"policy_{i}"] = agent_class(model_name=model_name, game_name=game_name)  # Load LLMS into GPU memory
        elif agent_type == "random":
            policies[f"policy_{i}"] = agent_class(seed=seed)
        elif agent_type == "human":
            policies[f"policy_{i}"] = agent_class()

        logger.info(f"Assigned: policy_{i} -> {agent_type.upper()} ({agent_config.get('model', 'N/A')})")

    return policies

def policy_mapping_fn(agent_id: str) -> str:
    """
    Maps an agent ID to a policy key.

    Args:
        agent_id (str): The agent's identifier.

    Returns:
        str: The corresponding policy key (e.g., "policy_0").
    """
    agent_id_str = str(agent_id)
    index = agent_id_str.split("_")[-1]
    policy_key = f"policy_{index}"
    logger.debug(f"Mapping agent {agent_id_str} -> {policy_key}")
    return policy_key
