"""Simulator for Matching Pennies (3-player).

This module implements the MatchingPenniesSimulator class, which simulates games of
Matching Pennies with three players using the OpenSpiel framework.
"""


from typing import Any, Dict
from envs.open_spiel_env import OpenSpielEnv

class MatchingPenniesSimulator(OpenSpielEnv):
    """Environment Simulator for Matching Pennies."""

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