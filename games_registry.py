# games_registry.py
"""Centralized registry for available games and their simulators."""

from games.tic_tac_toe import get_tic_tac_toe_game
from games.prisoners_dilemma import get_prisoners_dilemma_game
from games.connect_four import get_connect_four_game
from games.matrix_rps import get_matrix_rps_game
from games.matrix_pd import get_matrix_pd_game
from games.kuhn_poker import get_kuhn_poker_game
from games.matching_pennies import get_matching_pennies_game

from simulators.tic_tac_toe_simulator import TicTacToeSimulator
from simulators.prisoners_dilemma_simulator import PrisonersDilemmaSimulator
from simulators.connect_four_simulator import ConnectFourSimulator
from simulators.matrix_game_simulator import MatrixGameSimulator
from simulators.kuhn_poker_simulator import KuhnPokerSimulator
from simulators.matching_pennies_simulator import MatchingPenniesSimulator

# Register games here
GAMES_REGISTRY = {
    "tic_tac_toe": {
        "loader": get_tic_tac_toe_game,
        "simulator": TicTacToeSimulator,
        "display_name": "Tic-Tac-Toe",
    },
    "prisoners_dilemma": {
        "loader": get_prisoners_dilemma_game, # Iterated PD: Multi-round, emphasizing strategy development over repeated interactions.
        "simulator": PrisonersDilemmaSimulator,
        "display_name": "Iterated Prisoner's Dilemma",
    },
    "connect_four": {
        "loader": get_connect_four_game,
        "simulator": ConnectFourSimulator,
        "display_name": "Connect Four",
    },
    "rps": {
        "loader": get_matrix_rps_game,
        "simulator": MatrixGameSimulator,
        "display_name": "Rock-Paper-Scissors (Matrix)",
    },
    "matrix_pd": {
        "loader": get_matrix_pd_game,  # Matrix PD: Single-round, with the payoff matrix as the primary structure.
        "simulator": MatrixGameSimulator,
        "display_name": "Prisoner's Dilemma (Matrix)",
    },
    "kuhn_poker": {
        "loader": get_kuhn_poker_game,
        "simulator": KuhnPokerSimulator,
        "display_name": "Kuhn Poker",
    },
    "matching_pennies": {
        "loader": get_matching_pennies_game,
        "simulator": MatchingPenniesSimulator,
        "display_name": "Matching Pennies (3P)",
    },
}
