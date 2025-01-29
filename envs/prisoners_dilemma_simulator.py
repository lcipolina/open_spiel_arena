"""Simulator for Iterated Prisoner's Dilemma.

This module implements the PrisonersDilemmaSimulator class, which simulates games of
the Iterated Prisoner's Dilemma using the OpenSpiel framework.
"""
from typing import Any, Dict
import random

'''
class PrisonersDilemmaSimulator(GameSimulator):
    """Simulator for the Iterated Prisoner's Dilemma with stochastic termination.

    This class extends the base GameSimulator to handle the specific logic
    for chance nodes in the Iterated Prisoner's Dilemma, where the game
    terminates probabilistically after each round.
    """

    def _handle_chance_node(self, state: Any):
        """Handle the chance node for stochastic game termination.

        At each chance node, the game decides whether to continue or stop
        based on the termination probability defined in the game parameters.

        Args:
            state (pyspiel.State): The current state of the game.

        Raises:
            ValueError: If the chance outcomes are invalid.
        """
        outcomes, probabilities = zip(*state.chance_outcomes())
        sampled_outcome = random.choices(outcomes, probabilities)[0]
        state.apply_action(sampled_outcome)
'''


from typing import Any, Dict
from envs.open_spiel_env import OpenSpielEnv

class PrisonersDilemmaSimulator(OpenSpielEnv):
    """Simulator for the Iterated Prisoner's Dilemma with stochastic termination."""

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