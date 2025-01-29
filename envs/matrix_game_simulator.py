"""Simulator for Matrix Games.

This module implements the MatrixGameSimulator class, which handles various
matrix games like Rock-Paper-Scissors and Prisoner's Dilemma using the OpenSpiel
framework.
"""

from typing import Any, Dict
from envs.open_spiel_env import OpenSpielEnv

class MatrixGameSimulator(OpenSpielEnv):
    """Environment Simulator for Matrix Games."""

    def __init__(self, game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None):
        """
        Args:
            game: The OpenSpiel game object.
            game_name: A string representing the name of the game.
            player_types: A dictionary mapping player IDs to their types (e.g., human, random).
            max_game_rounds: Maximum number of rounds
                             for iterated games (optional, default is None).
        """
        super().__init__(game, game_name, player_types, max_game_rounds)