"""A module for loading and handling the Rock-Paper-Scissors game logic.

This module uses OpenSpiel's implementation of the Rock-Paper-Scissors game. It
provides utility functions for game setup and management, allowing the simulation
framework to play games of Rock-Paper-Scissors.
"""

import pyspiel

def get_rps_game():
    """Load the Rock-Paper-Scissors game instance.

    Returns:
        pyspiel.Game: The Rock-Paper-Scissors game instance from OpenSpiel.
    """
    return pyspiel.load_game("matrix_rps")

