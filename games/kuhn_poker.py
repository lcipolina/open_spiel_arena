"""A module for loading and handling the Kuhn Poker game logic.

This module uses OpenSpiel's implementation of the Kuhn Poker game. It provides
utility functions to set up and retrieve the game instance, which can be used by
the simulation framework to simulate gameplay.
"""

import pyspiel

def get_kuhn_poker_game():
    """Load the Kuhn Poker game instance.

    Returns:
        pyspiel.Game: The Kuhn Poker game instance from OpenSpiel.
    """
    return pyspiel.load_game("kuhn_poker")
