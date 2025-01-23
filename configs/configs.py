"""
configs.py - Simple configuration system with JSON and key-value CLI overrides.
"""

import argparse
import json
from typing import Any, Dict


def default_simulation_config() -> Dict[str, Any]:
    """Returns the default simulation configuration."""
    return {
        "env_config": {
            "game_name": "tic_tac_toe",
            "max_game_rounds": 1,
        },
        "num_episodes": 5,
        "seed": 42,
        "agents": [
            {"type": "human", "name": "Player 1"},
            {"type": "random", "name": "Player 2"},
        ],
        "alternate_first_player": True,
        "log_level": "INFO",
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """Creates a simple CLI parser for key-value overrides and JSON config."""
    parser = argparse.ArgumentParser(
        description="Simulation Configuration",
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
    config = default_simulation_config()

    # Load JSON config
    if args.config:
        if args.config.strip().startswith("{"):
            # Raw JSON string
            json_config = json.loads(args.config)
        else:
            # JSON file
            with open(args.config, "r") as f:
                json_config = json.load(f)
        config.update(json_config)

    # Apply key-value overrides
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            config = apply_override(config, key, value)

    return config


def apply_override(config: Dict[str, Any], key: str, value: str) -> Dict[str, Any]:
    """Applies a key-value override to the configuration."""
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    # Convert value to correct type
    current[keys[-1]] = parse_value(value)
    return config


def parse_value(value: str) -> Any:
    """Converts a string value to the appropriate type (int, float, bool, etc.)."""
    try:
        return json.loads(value)  # Automatically parses JSON types
    except json.JSONDecodeError:
        return value  # Leave as string if parsing fails
