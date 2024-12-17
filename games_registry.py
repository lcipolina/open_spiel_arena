# games_registry.py
"""Centralized registry for available games and their simulators."""

from games.tic_tac_toe import get_tic_tac_toe_game
from games.prisoners_dilemma import get_prisoners_dilemma_game
from games.rock_paper_scissors import get_rps_game
from simulators.tic_tac_toe_simulator import TicTacToeSimulator
from simulators.prisoners_dilemma_simulator import PrisonersDilemmaSimulator
from simulators.rock_paper_scissors_simulator import RockPaperScissorsSimulator

# Register games here
GAMES_REGISTRY = {
    "tic_tac_toe": {
        "loader": get_tic_tac_toe_game,
        "simulator": TicTacToeSimulator,
        "display_name": "Tic-Tac-Toe",
    },
    "prisoners_dilemma": {
        "loader": get_prisoners_dilemma_game,
        "simulator": PrisonersDilemmaSimulator,
        "display_name": "Iterated Prisoner's Dilemma",
    },
    "rps": {
        "loader": get_rps_game,
        "simulator": RockPaperScissorsSimulator,
        "display_name": "Rock-Paper-Scissors",
    },
}
