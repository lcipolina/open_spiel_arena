"""
Logging utilities for tracking simulations, agent behavior, and game outcomes.
Supports structured logging and experiment tracking.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import traceback
from logging.handlers import RotatingFileHandler

import json
from typing import List, Dict, Any
from collections import defaultdict


def generate_game_log(
    model_name: str,
    games: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates a structured game log containing details of individual games and a summary.

    Args:
        model_name: The name of the model that played the games.
        games: A list of game records, where each record contains details such as
               game name, rounds, result, moves, illegal moves, and opponent.

    Returns:
        Dict[str, Any]: A dictionary containing structured game logs and summary statistics.
    """

    # Initialize summary statistics
    summary = defaultdict(lambda: {
        "games": 0,
        "moves/game": 0.0,
        "illegal-moves": 0,
        "win-rate": 0.0,
        "vs Random": 0.0
    })

    # Track per-game statistics
    game_stats = defaultdict(lambda: {
        "total_moves": 0,
        "illegal_moves": 0,
        "wins": 0,
        "vs_random_wins": 0,
        "vs_random_games": 0
    })

    for game in games:
        game_name = game["game"]
        rounds = game["rounds"]
        result = game["result"]
        moves = game["moves"]
        illegal_moves = game["illegal_moves"]
        opponent = game["opponent"]

        # Update game-specific statistics
        game_stats[game_name]["total_moves"] += len(moves)
        game_stats[game_name]["illegal_moves"] += illegal_moves
        game_stats[game_name]["wins"] += int(result == "win")
        game_stats[game_name]["vs_random_wins"] += int(
            result == "win" and opponent == "random_bot"
        )
        game_stats[game_name]["vs_random_games"] += int(opponent == "random_bot")
        summary[game_name]["games"] += 1

    # Compute final summary statistics
    for game_name, stats in game_stats.items():
        total_games = summary[game_name]["games"]
        summary[game_name]["moves/game"] = round(
            stats["total_moves"] / total_games, 1
        )
        summary[game_name]["illegal-moves"] = stats["illegal_moves"]
        summary[game_name]["win-rate"] = round(
            (stats["wins"] / total_games) * 100, 1
        ) if total_games > 0 else 0.0
        if stats["vs_random_games"] > 0:
            summary[game_name]["vs Random"] = round(
                (stats["vs_random_wins"] / stats["vs_random_games"]) * 100, 1
            )
        else:
            summary[game_name]["vs Random"] = 0.0

    return {
        "model_name": model_name,
        "games_played": games,
        "summary": summary
    }


def save_game_log(file_path: str, game_log: Dict[str, Any]):
    """
    Saves the game log to a JSON file.

    Args:
        file_path: The file path where the log should be saved.
        game_log: The structured game log dictionary.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(game_log, f, indent=4)


def load_game_log(file_path: str) -> Dict[str, Any]:
    """
    Loads a game log from a JSON file.

    Args:
        file_path: The file path of the log file.

    Returns:
        Dict[str, Any]: The loaded game log.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class SimulationFormatter(logging.Formatter):
    """Custom formatter for game simulation records"""
    def format(self, record):
        if hasattr(record, 'game_state'):
            record.msg = f"[Game {record.game_name}] {record.msg}"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """Structured JSON logging for analytics"""
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add custom fields
        for attr in ["game", "player", "action", "reward"]:
            if hasattr(record, attr):
                log_record[attr] = getattr(record, attr)

        if record.exc_info:
            log_record["exception"] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_record)

def configure_logging(
    log_dir: str = "logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    max_log_size: int = 10  # MB
):
    """
    Configure global logging infrastructure

    Args:
        log_dir: Directory to store log files
        console_level: Log level for console output
        file_level: Log level for file output
        max_log_size: Maximum log file size in MB before rotation
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = SimulationFormatter(
        "%(asctime)s - %(levelname)s - %(game_name)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        filename=Path(log_dir) / "simulations.log",
        maxBytes=max_log_size * 1024 * 1024,
        backupCount=5
    )
    file_handler.setLevel(file_level)
    file_formatter = JSONFormatter()
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

def log_game_event(
    game_name: str,
    event_type: str,
    details: Dict[str, Any],
    level: str = "INFO"
):
    """
    Structured logging for game events

    Example:
        log_game_event(
            game_name="tic_tac_toe",
            event_type="move",
            details={
                "player": "LLM Agent",
                "action": 4,
                "state": "x.o\n...\n..."
            }
        )
    """
    logger = logging.getLogger("game_events")
    extra = {"game": game_name, **details}

    if level == "DEBUG":
        logger.debug(json.dumps({"event_type": event_type, **details}), extra=extra)
    elif level == "WARNING":
        logger.warning(json.dumps({"event_type": event_type, **details}), extra=extra)
    else:
        logger.info(json.dumps({"event_type": event_type, **details}), extra=extra)

def log_experiment_config(config: Dict[str, Any]):
    """Log experiment configuration with metadata"""
    logger = logging.getLogger("experiments")
    logger.info(
        "Experiment Configuration",
        extra={
            "config": config,
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
    )
