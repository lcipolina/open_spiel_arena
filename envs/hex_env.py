"""Simulator for Hex.

This module implements the TicTacToeSimulator class, which simulates games of
Hex using the OpenSpiel framework.
"""

from typing import Any, Dict, Optional
from envs.open_spiel_env import OpenSpielEnv

class HexEnv(OpenSpielEnv):
    """Environment Simulator for Hex."""

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
        """Describes legal actions as indices on a hex board.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: Legal action numbers and a flattened board index layout.
        """
        legal = self.state.legal_actions(agent_id)
        size = self.game.board_size  # Usually 11

        # Create a flat index grid (diagonal shape)
        grid = []
        idx = 0
        for r in range(size):
            indent = "  " * r  # diagonal effect
            row = " ".join(f"{idx + c:2}" for c in range(size))
            grid.append(f"{indent}{row}")
            idx += size

        index_grid = "\n".join(grid)
        return f"{legal} (board indices)\n\nBoard index layout:\n{index_grid}"

    def render_board_with_indices(self, agent_id: int) -> str:
        """Renders Hex board showing moves and legal action indices.

        Args:
            agent_id (int): The player's ID.

        Returns:
            str: A hex-aligned board with pieces and index references.
        """
        legal = set(self.state.legal_actions(agent_id))
        tensor = self.state.observation_tensor(agent_id)
        size = self.game.board_size  # e.g., 11

        rows = []
        for row in range(size):
            indent = "  " * row
            row_cells = []
            for col in range(size):
                idx = row * size + col
                offset = idx * 3
                if tensor[offset] == 1:  # Player 0 (y)
                    row_cells.append("y")
                elif tensor[offset + 1] == 1:  # Player 1 (o)
                    row_cells.append("o")
                elif idx in legal:
                    row_cells.append(f"{idx}")
                else:
                    row_cells.append(".")
            rows.append(indent + " ".join(f"{c:>2}" for c in row_cells))
        return "\n".join(rows)
