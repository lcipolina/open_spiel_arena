"""
configs.py - Simple configuration system with JSON and key-value CLI overrides.
"""

import argparse
import json
from typing import Any, Dict
from games.registry import registry # Initilizes an empty registry dictionary


def default_simulation_config() -> Dict[str, Any]:
    """Returns the default simulation configuration for multiple games."""
    return {
        "env_configs": [  # A list of game-specific configurations
            # {
            #    "game_name": "kuhn_poker",  # ok
            #     "max_game_rounds": None,  # Only use this for iterated games
            # # Additional game-specific parameters
           # },
             {
                "game_name": "connect_four",
                "max_game_rounds": None,  # Only use this for iterated games
            },
        #    {
        #         "game_name": "tic_tac_toe",  #ok
        #         "max_game_rounds": None,  # Only use this for iterated games
        #     },
           #     {
           #     "game_name": "matching_pennies",
           #     "max_game_rounds": None,  # Only use this for iterated games
          #  },
        ],
        "num_episodes": 3,  #TODO: MORE THAN # IT COLLAPSES!!
        "seed": 42,
        "use_ray": False,
        "mode": "llm_vs_random",  # "manual", "llm_vs_llm"
        "agents": {
            # Define agent settings (shared across games)
        },
        "log_level": "INFO",
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """Creates a simple CLI parser for key-value overrides and JSON config."""
    parser = argparse.ArgumentParser(
        description="Game Simulation Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON config file or raw JSON string.",
    )

    parser.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        help="Key-value overrides for configuration (e.g., game_name=tic_tac_toe).",
    )
    return parser


def parse_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Parses the configuration, merging JSON config and CLI overrides."""

    # Default config
    config = default_simulation_config()

    # Update with JSON config (if provided)
    if args.config:
        if args.config.strip().startswith("{"):
            # Raw JSON string
            json_config = json.loads(args.config)
        else:
            # JSON file
            with open(args.config, "r") as f:
                json_config = json.load(f)
        config.update(json_config)

    # Apply CLI key-value overrides (if provided)
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            config = apply_override(config, key, value)

    return config


def apply_override(config: Dict[str, Any], key: str, value: str) -> Dict[str, Any]:
    """Applies a key-value override to the configuration."""
    keys = key.split(".")
    current = config

    for i, k in enumerate(keys[:-1]):
        # Handle dictionary keys
        if k.isdigit():
            k = int(k)  # Convert index to integer
            if not isinstance(current, dict) or k not in current:
                raise ValueError(f"Invalid key '{k}' in override '{key}'")
        current = current.setdefault(k, {}) # type: ignore

    # Handle the final key
    final_key = keys[-1]
    if final_key.isdigit():
        final_key = int(final_key)
        if not isinstance(current, dict) or final_key not in current:
            raise ValueError(f"Invalid key '{final_key}' in override '{key}'")
    current[final_key] = parse_value(value)  # type: ignore

    return config


def parse_value(value: str) -> Any:
    """Converts a string value to the appropriate type (int, float, bool, etc.)."""
    try:
        return json.loads(value)  # Automatically parses JSON types
    except json.JSONDecodeError:
        return value  # Leave as string if parsing fails


def validate_config(config: Dict[str, Any]) -> None:
    """Validates the configuration."""
    game_name = config["env_config"]["game_name"]
    num_players = registry.get_game_loader(game_name)().num_players()

    # Check if the number of agents matches the number of players
    if len(config["agents"]) != num_players:
        raise ValueError(
            f"Game '{game_name}' requires {num_players} players, "
            f"but {len(config['agents'])} agents were provided."
        )

    # Validate agent types
    valid_agent_types = {"human", "random", "llm"}
    for player, agent in config["agents"].items():
        if agent["type"] not in valid_agent_types:
            raise ValueError(f"Invalid agent type for {player}: {agent['type']} (must be one of {valid_agent_types})")
        if agent["type"] == "llm" and not agent.get("model"):
            raise ValueError(f"LLM agent '{player}' must specify a model (e.g., gpt-4)")
