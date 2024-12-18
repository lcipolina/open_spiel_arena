"""A module for loading and handling the Matrix Rock-Paper-Scissors game logic.

This module uses OpenSpiel's implementation of the Matrix Rock-Paper-Scissors (RPS)
game. It provides utility functions to set up and retrieve the game instance, which
can be used by the simulation framework to simulate gameplay.
"""

import pyspiel

def get_matrix_rps_game():
    """Load the Matrix Rock-Paper-Scissors game instance.

    Returns:
        pyspiel.Game: The Matrix RPS game instance from OpenSpiel.
    """
    return pyspiel.load_game("matrix_rps")
