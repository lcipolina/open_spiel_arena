"""A module for loading and handling the Matching Pennies (3-player) game logic.

This module uses OpenSpiel's implementation of the Matching Pennies (3-player) game.
It provides utility functions to set up and retrieve the game instance, which can be
used by the simulation framework to simulate gameplay.
"""

import pyspiel

def get_matching_pennies_game():
    """Load the Matching Pennies (3-player) game instance.

    Returns:
        pyspiel.Game: The Matching Pennies (3-player) game instance from OpenSpiel.
    """
    return pyspiel.load_game("matching_pennies_3p")
