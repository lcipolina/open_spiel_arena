"""Simulator for Iterated Prisoner's Dilemma.

This module implements the PrisonersDilemmaSimulator class, which simulates games of
the Iterated Prisoner's Dilemma using the OpenSpiel framework.
"""
from typing import Any, Dict

from simulators.base_simulator import GameSimulator

class PrisonersDilemmaSimulator(GameSimulator):
    """Simulator for Iterated Prisoner's Dilemma."""

    def __init__(self, game: Any, game_name: str, llms: Dict[str, Any],
                 random_bot: bool = False, play_against_itself: bool = False,
                 max_iterations: int = 50) -> None:
        super().__init__(game, game_name, llms, random_bot, play_against_itself)
        self.max_iterations = max_iterations
