"""A module for loading and handling the Matrix Prisoner's Dilemma game logic.

This module uses OpenSpiel's implementation of the Matrix Prisoner's Dilemma (PD)
game. It provides utility functions to set up and retrieve the game instance, which
can be used by the simulation framework to simulate gameplay.
"""

import pyspiel

def get_matrix_pd_game():
    """Load the Matrix Prisoner's Dilemma game instance.

    Returns:
        pyspiel.Game: The Matrix PD game instance from OpenSpiel.
    """
    return pyspiel.load_game("matrix_pd")
