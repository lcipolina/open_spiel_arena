# utils/common_utils.py
"""Common utility functions for the OpenSpiel LLM Arena project.

This module provides shared utility functions for logging, configuration,
and other cross-cutting concerns.
"""

import logging

def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger for the simulation.

    Args:
        name: The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

