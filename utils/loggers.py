"""
Logging utilities for tracking simulations, agent behavior, and game outcomes.
Supports structured logging and experiment tracking.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Any, List
import traceback
from logging.handlers import RotatingFileHandler
import os
import time
from functools import wraps
from collections import defaultdict

from utils.results_utils import print_total_scores

def generate_game_log(model_name: str, games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates a structured game log containing details of individual games and a summary.

    Args:
        model_name: The name of the model that played the games.
        games: A list of game records.

    Returns:
        Dict[str, Any]: A dictionary containing structured game logs and summary statistics.
    """

    summary = {}

    # Track LLM wins per model and per opponent type
    llm_win_counts = {}  # {model_name: win_count}
    llm_total_games = {}  # {model_name: total_games_played}
    llm_vs_opponent = {}  # {model_name: {opponent_type: win_count}}

    for game in games:
        game_name = game["game"]
        rounds = game["rounds"]
        moves = game["moves"]
        illegal_moves = game["illegal_moves"]

        if game_name not in summary:
            summary[game_name] = {
                "games": 0,
                "moves/game": 0.0,
                "illegal-moves": 0,
                "win-rate": {},
                "win-rate vs": {}  # 🔹 Track win rates per opponent type
            }

        summary[game_name]["games"] += 1
        summary[game_name]["moves/game"] += len(moves) / summary[game_name]["games"]
        summary[game_name]["illegal-moves"] += illegal_moves

        # Identify LLMs and opponents
        llm_players = [p for p in game["players"] if p["player_type"] == "llm"]
        non_llm_players = [p for p in game["players"] if p["player_type"] != "llm"]

        for llm in llm_players:
            model = llm["player_model"]
            result = llm["result"]

            # Initialize model tracking
            if model not in llm_win_counts:
                llm_win_counts[model] = 0
                llm_total_games[model] = 0
                llm_vs_opponent[model] = {}

            llm_total_games[model] += 1  # Count games played by this LLM

            if result == "win":
                llm_win_counts[model] += 1  # Count wins

                # Track wins against each opponent type
                for opponent in non_llm_players:
                    opponent_type = opponent["player_type"]
                    if opponent_type not in llm_vs_opponent[model]:
                        llm_vs_opponent[model][opponent_type] = 0
                    llm_vs_opponent[model][opponent_type] += 1

    # 🔹 Compute win rates per LLM model and opponent type
    for model, total_games in llm_total_games.items():
        win_rate = (llm_win_counts[model] / total_games) * 100 if total_games > 0 else 0
        summary[game_name]["win-rate"][model] = round(win_rate, 2)

        # Compute per-opponent win rates
        for opponent_type, wins in llm_vs_opponent[model].items():
            total_vs_opponent = sum(1 for game in games for p in game["players"]
                                    if p["player_type"] == opponent_type and p["player_model"] == model)

            win_rate_vs_opponent = (wins / total_vs_opponent) * 100 if total_vs_opponent > 0 else 0
            summary[game_name]["win-rate vs"][f"{model} vs {opponent_type}"] = round(win_rate_vs_opponent, 2)

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

#TODO: delete this function!
def configure_logging_log(
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


def time_execution(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Args:
        func: Function to be wrapped.

    Returns:
        Wrapped function with execution time logging.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        return result
    return wrapper


def log_simulation_results(func: Callable) -> Callable:
    """
    Decorator to log and save game simulation results.

    Args:
        func: The simulation function.

    Returns:
        Wrapped function that logs results.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Expected (model_name, game_results) from simulation function.")

        model_name, game_results = result
        game_name = game_results[0]["game"] if game_results else "unknown_game"

        # Generate structured game log
        log_data = generate_game_log(model_name, game_results)

        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_path = f"results/{game_name}_{timestamp}.json"

        save_game_log(file_path, log_data)

        # Print summary to console
        print_total_scores(game_name, log_data["summary"])

        logging.info(f"Simulation complete. Results saved to {file_path}")

        return result  # Return the original function output

    return wrapper
