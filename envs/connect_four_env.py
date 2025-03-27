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

    def render_board_with_indices(self, agent_id: int) -> str:
        """Renders Connect Four board with checker positions and column indices.

        Args:
            agent_id (int): The player's ID (ignored here).

        Returns:
            str: A grid showing board state and column numbers for LLM clarity.
        """
        # Assume board is 6 rows (bottom to top), 7 columns
        rows = []
        tensor = self.state.observation_tensor(agent_id)
        # OpenSpiel stores board as one-hot per cell: [player0, player1, empty]
        num_rows, num_cols = 6, 7

        for row in reversed(range(num_rows)):
            row_str = []
            for col in range(num_cols):
                idx = (row * num_cols + col) * 3
                if tensor[idx] == 1:  # Player 0
                    row_str.append("x")
                elif tensor[idx + 1] == 1:  # Player 1
                    row_str.append("o")
                else:
                    row_str.append(".")
            rows.append(" " + " | ".join(row_str))

        grid = "\n" + "\n" + "\n".join(rows)
        col_indices = " " + "   ".join(str(c) for c in range(num_cols))
        return f"{grid}\n\nColumn indices:\n{col_indices}"

