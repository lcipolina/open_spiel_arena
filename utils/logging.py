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
