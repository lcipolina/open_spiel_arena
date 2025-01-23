"""
loaders_module.py

A single module containing loader functions for all games.
Used by the 'games_registry.py' module.
"""

import pyspiel


def get_prisoners_dilemma_game():
    """
    Load the Python Iterated Prisoner's Dilemma game instance.
    Returns:
        pyspiel.Game: The Iterated Prisoner's Dilemma game instance from OpenSpiel.
    """
    return pyspiel.load_game("python_iterated_prisoners_dilemma")


def get_tic_tac_toe_game():
    """
    Load the Tic-Tac-Toe game instance.
    Returns:
        TicTacToeGame: The Tic-Tac-Toe game instance from OpenSpiel.
    """
    return pyspiel.load_game("tic_tac_toe")


def get_connect_four_game():
    """Load the Connect Four game instance.

    Returns:
        pyspiel.Game: The Connect Four game instance from OpenSpiel.
    """
    return pyspiel.load_game("connect_four")


def get_kuhn_poker_game():
    """Load the Kuhn Poker game instance.

    Returns:
        pyspiel.Game: The Kuhn Poker game instance from OpenSpiel.
    """
    return pyspiel.load_game("kuhn_poker")


def get_matching_pennies_game():
    """Load the Matching Pennies (3-player) game instance.

    Returns:
        pyspiel.Game: The Matching Pennies (3-player) game instance from OpenSpiel.
    """
    return pyspiel.load_game("matching_pennies_3p")


def get_matrix_pd_game():
    """Load the Matrix Prisoner's Dilemma game instance.

    Returns:
        pyspiel.Game: The Matrix PD game instance from OpenSpiel.
    """
    return pyspiel.load_game("matrix_pd")


def get_matrix_rps_game():
    """Load the Matrix Rock-Paper-Scissors game instance.

    Returns:
        pyspiel.Game: The Matrix RPS game instance from OpenSpiel.
    """
    return pyspiel.load_game("matrix_rps")
