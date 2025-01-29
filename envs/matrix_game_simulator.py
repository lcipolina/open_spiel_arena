"""Simulator for Matrix Games.

This module implements the MatrixGameSimulator class, which handles various
matrix games like Rock-Paper-Scissors and Prisoner's Dilemma using the OpenSpiel
framework.
"""

from typing import Any, Dict, List
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


    def apply_action(self, action: List[int]):
        """Applies the given list of actions to the environment.

        Args:
            action: If the game is simultaneous-move,
                it is a list of actions (one for each player).
        """
        self.state.apply_actions(action)

    def _state_to_observation(self) -> Dict[str, Any]:
        """
        Generate the observation for Rock-Paper-Scissors.

        Returns:
            Dict[str, Any]: Observation dictionary containing:
                - state_string: A placeholder for state description (None in RPS).
                - legal_actions: A list of valid actions for each player.
                - info: A string providing action descriptions.
        """

        # TODO: confirm why legal actions come as [0,1,2,3] instead of [0,1]

        row_action_ids = self.state.legal_actions(0)  # Get action indices (0,1)
        row_action_names = [self.state.action_to_string(0, action) for action in row_action_ids]

        return {
            "state_string": None,  # No meaningful observation string
            "legal_actions": [row_action_ids for _ in range(self.state.num_players())],
            "info": f"Actions: {row_action_names}"
        }