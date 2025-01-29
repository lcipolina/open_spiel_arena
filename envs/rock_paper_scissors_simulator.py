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

    '''
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
    '''

    def _state_to_observation(self) -> Dict[str, Any]:
        """
        Generate the observation for Rock-Paper-Scissors.

        Returns:
            Dict[str, Any]: Observation dictionary containing:
                - state_string: A placeholder for state description (None in RPS).
                - legal_actions: A list of valid actions for each player.
                - info: A string providing action descriptions.
        """
        return {
            "state_string": None,  # RPS has no meaningful observation string
            "legal_actions": [[0, 1, 2] for _ in range(self.state.num_players())],
            "info": "Actions: Rock (0), Paper (1), Scissors (2)"  # Fixed string formatting
        }
