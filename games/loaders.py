"""
/games/loaders.py

Centralized game loader module with decorator-based registration.
"""

import pyspiel
from games.registry import registry


class GameLoader:
    """Base class for game loaders"""

# Registering games with simplified registry format
@registry.register(
    name="prisoners_dilemma",
    module_path="games.loaders",
    class_name="PrisonersDilemmaLoader",
    simulator_path="envs.prisoners_dilemma_simulator.PrisonersDilemmaSimulator",
    display_name="Iterated Prisoner's Dilemma"
)
class PrisonersDilemmaLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("python_iterated_prisoners_dilemma")

@registry.register(
    name="tic_tac_toe",
    module_path="games.loaders",
    class_name="TicTacToeLoader",
    simulator_path="envs.tic_tac_toe_simulator.TicTacToeSimulator",
    display_name="Tic-Tac-Toe"
)
class TicTacToeLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("tic_tac_toe")

@registry.register(
    name="connect_four",
    module_path="games.loaders",
    class_name="ConnectFourLoader",
    simulator_path="envs.connect_four_simulator.ConnectFourSimulator",
    display_name="Connect Four"
)
class ConnectFourLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("connect_four")

@registry.register(
    name="kuhn_poker",
    module_path="games.loaders",
    class_name="KuhnPokerLoader",
    simulator_path="envs.kuhn_poker_simulator.KuhnPokerSimulator",
    display_name="Kuhn Poker"
)
class KuhnPokerLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("kuhn_poker")


# Prisoner's Dilemma, Matching Pennies and Rock-Paper-Scissors as matrix games

@registry.register(
    name="matrix_pd",
    module_path="games.loaders",
    class_name="MatrixPDLoader",
    simulator_path="envs.matrix_game_simulator.MatrixGameSimulator",
    display_name="Matrix Prisoner's Dilemma"
)
class MatrixPDLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matrix_pd")

@registry.register(
    name="matching_pennies",
    module_path="games.loaders", #pyspiel loader
    class_name="MatchingPenniesLoader",
    simulator_path="envs.matrix_game_simulator.MatrixGameSimulator", # environment
    display_name="Matching Pennies (3P)"
)
class MatchingPenniesLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matching_pennies_3p")

@registry.register(
    name="matrix_rps",
    module_path="games.loaders",
    class_name="MatrixRPSLoader",
    simulator_path="envs.matrix_game_simulator.MatrixGameSimulator",
    display_name="Matrix Rock-Paper-Scissors"
)
class MatrixRPSLoader(GameLoader):
    @staticmethod
    def load():
        return pyspiel.load_game("matrix_rps")
