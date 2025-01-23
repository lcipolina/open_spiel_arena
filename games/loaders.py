"""
/games/loaders.py

Centralized game loader module with decorator-based registration.
"""

import pyspiel


import pyspiel
from .registry import registry

class GameLoader:
    """Base class for game loaders (keeps registry clean)"""

@registry.register(
    name="prisoners_dilemma",
    loader_path="openspiel_llm.games.loaders.PrisonersDilemmaLoader.load",
    simulator_path="openspiel_llm.simulators.prisoners_dilemma.PrisonersDilemmaSimulator",
    display_name="Iterated Prisoner's Dilemma"
)
class PrisonersDilemmaLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("python_iterated_prisoners_dilemma")

@registry.register(
    name="tic_tac_toe",
    loader_path="openspiel_llm.games.loaders.TicTacToeLoader.load",
    simulator_path="openspiel_llm.simulators.tic_tac_toe.TicTacToeSimulator",
    display_name="Tic-Tac-Toe"
)
class TicTacToeLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("tic_tac_toe")

@registry.register(
    name="connect_four",
    loader_path="openspiel_llm.games.loaders.ConnectFourLoader.load",
    simulator_path="openspiel_llm.simulators.connect_four.ConnectFourSimulator",
    display_name="Connect Four"
)
class ConnectFourLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("connect_four")

@registry.register(
    name="kuhn_poker",
    loader_path="openspiel_llm.games.loaders.KuhnPokerLoader.load",
    simulator_path="openspiel_llm.simulators.kuhn_poker.KuhnPokerSimulator",
    display_name="Kuhn Poker"
)
class KuhnPokerLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("kuhn_poker")

@registry.register(
    name="matching_pennies",
    loader_path="openspiel_llm.games.loaders.MatchingPenniesLoader.load",
    simulator_path="openspiel_llm.simulators.matching_pennies.MatchingPenniesSimulator",
    display_name="Matching Pennies (3P)"
)
class MatchingPenniesLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matching_pennies_3p")

@registry.register(
    name="matrix_pd",
    loader_path="openspiel_llm.games.loaders.MatrixPDLoader.load",
    simulator_path="openspiel_llm.simulators.matrix_games.MatrixPDSimulator",
    display_name="Matrix Prisoner's Dilemma"
)
class MatrixPDLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matrix_pd")

@registry.register(
    name="matrix_rps",
    loader_path="openspiel_llm.games.loaders.MatrixRPSLoader.load",
    simulator_path="openspiel_llm.simulators.matrix_games.MatrixRPSSimulator",
    display_name="Matrix Rock-Paper-Scissors"
)
class MatrixRPSLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matrix_rps")