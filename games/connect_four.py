"""A module for loading and handling the Connect Four game logic.

This module uses OpenSpiel's implementation of the Connect Four game. It provides
utility functions to set up and retrieve the game instance, which can be used by
the simulation framework to simulate gameplay.
"""

import pyspiel

def get_connect_four_game():
    """Load the Connect Four game instance.

    Returns:
        pyspiel.Game: The Connect Four game instance from OpenSpiel.
    """
    return pyspiel.load_game("connect_four")
