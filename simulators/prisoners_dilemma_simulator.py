"""Simulator for Iterated Prisoner's Dilemma.

This module implements the PrisonersDilemmaSimulator class, which simulates games of
the Iterated Prisoner's Dilemma using the OpenSpiel framework.
"""
from typing import Any, Dict
import random

from simulators.base_simulator import GameSimulator


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
