"""A module for loading and handling Tic-Tac-Toe game logic using OpenSpiel.

This module interfaces with OpenSpiel's implementation of Tic-Tac-Toe and provides
utility functions for game setup and management. It is used by the simulation framework
to initialize and manage the Tic-Tac-Toe game instance.
"""

from open_spiel.python.games.tic_tac_toe import TicTacToeGame

def get_tic_tac_toe_game():
    """Load the Tic-Tac-Toe game instance.

    Returns:
        TicTacToeGame: The Tic-Tac-Toe game instance from OpenSpiel.
    """
    return TicTacToeGame()

