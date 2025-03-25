"""Simulator for Connect Four.

This module implements the ConnectFourEnv class, which simulates games of
Connect Four using the OpenSpiel framework.
"""

from typing import Any, Dict, Optional
from envs.open_spiel_env import OpenSpielEnv

class ConnectFourEnv(OpenSpielEnv):
    """Environment Simulator for Connect Four (turn based game)."""

    def __init__(self, game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None,
                 seed: Optional[int] = None):
        """
        Args:
            game: The OpenSpiel game object.
            game_name: A string representing the name of the game.
            player_types: A dictionary mapping player IDs to their types (e.g., human, random).
            max_game_rounds: Maximum number of rounds
                             for iterated games (optional, default is None).
        """
        super().__init__(game, game_name, player_types, max_game_rounds, seed)


    def get_player_symbol(self, agent_id: int) -> str:
        """Returns the symbol used by a Connect Four player.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: 'x' for player 0, 'o' for player 1.
        """
        return "x" if agent_id == 0 else "o"

    def describe_legal_actions(self, agent_id: int) -> str:
        """Returns the available column numbers for Connect Four.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A list of legal column indices to drop a checker.
        """
        legal = self.state.legal_actions(agent_id)
        return f"{legal} (column numbers)"
