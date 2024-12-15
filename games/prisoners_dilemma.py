"""A module for loading and handling the Iterated Prisoner's Dilemma game logic.

This module uses OpenSpiel's implementation of the Python Iterated Prisoner's Dilemma
game. It provides utility functions to set up and retrieve the game instance, which can
be used by the simulation framework to simulate gameplay.
"""

import pyspiel

def get_prisoners_dilemma_game():
    """Load the Python Iterated Prisoner's Dilemma game instance.

    Returns:
        pyspiel.Game: The Iterated Prisoner's Dilemma game instance from OpenSpiel.
    """
    return pyspiel.load_game("python_iterated_prisoners_dilemma")

