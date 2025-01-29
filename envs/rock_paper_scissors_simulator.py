"""Simulator for Rock-Paper-Scissors.

This module implements the RockPaperScissorsSimulator class, which simulates games of
Rock-Paper-Scissors using the OpenSpiel framework.
"""

from typing import Any, Dict
from envs.open_spiel_env import OpenSpielEnv

class RockPaperScissorsSimulator(OpenSpielEnv):
    """Environment Simulator for Rock-Paper-Scissors."""

    def __init__(self, game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None):
        """
        Args:
            game: The OpenSpiel game object.
            game_name: A string representing the name of the game.
            player_types: A dictionary mapping player IDs to their types (e.g., human, random).
            max_game_rounds: Maximum number of rounds for iterated games (optional, default is None).
        """
        super().__init__(game, game_name, player_types, max_game_rounds)

    def get_observation(self, state) -> Dict[str, Any]:
        """
        Generate the observation for RPS.

        Args:
            state: The current game state.

        Returns:
            Dict[str, Any]: Observation with legal actions for all players.
        """
        return {
            "state_string": None,  # RPS has no meaningful observation string
            "legal_actions": [
                [0, 1, 2] for _ in range(state.num_players())
            ]  # Actions: Rock (0), Paper (1), Scissors (2)
        }
