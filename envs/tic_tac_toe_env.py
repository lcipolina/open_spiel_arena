"""Simulator for Tic-Tac-Toe.

This module implements the TicTacToeEnv class, which simulates games of
Tic-Tac-Toe using the OpenSpiel framework.
"""

from typing import Any, Dict, Optional
from envs.open_spiel_env import OpenSpielEnv

class TicTacToeEnv(OpenSpielEnv):
    """Environment Simulator for Tic-Tac-Toe."""

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
        """Returns the symbol used by a Tic Tac Toe player.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: 'x' for player 0, 'o' for player 1.
        """
        return "x" if agent_id == 0 else "o"

    def describe_legal_actions(self, agent_id: int) -> str:
        """Describes legal actions as board positions in a 3x3 grid.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A list of legal action numbers with positional meaning.
        """
        legal = self.state.legal_actions(agent_id)
        mapping_grid = (
            " 0 | 1 | 2\n"
            "-----------\n"
            " 3 | 4 | 5\n"
            "-----------\n"
            " 6 | 7 | 8\n"
        )
        return f"{legal} (cell indices)\n\nCell layout:\n{mapping_grid}"

    def render_board_with_indices(self, agent_id: int) -> str:
        """Renders the board showing symbols and open cell indices.

        Args:
            agent_id (int): The player's ID (ignored here).

        Returns:
            str: A 3x3 board with current moves and legal move indices.
        """
        legal = set(self.state.legal_actions(agent_id))
        board = self.state.observation_tensor(agent_id)

        # OpenSpiel uses one-hot for each cell: x=0, o=1, empty=2
        rows = []
        for row in range(3):
            cells = []
            for col in range(3):
                idx = row * 3 + col
                offset = idx * 3
                if board[offset] == 1:  # Player 0 (x)
                    cells.append("x")
                elif board[offset + 1] == 1:  # Player 1 (o)
                    cells.append("o")
                else:
                    cells.append(str(idx) if idx in legal else ".")
            rows.append(" " + " | ".join(cells))
        return "\n-----------\n".join(rows)
