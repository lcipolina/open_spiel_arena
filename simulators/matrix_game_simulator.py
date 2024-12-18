"""Simulator for Matrix Games.

This module implements the MatrixGameSimulator class, which handles various
matrix games like Rock-Paper-Scissors and Prisoner's Dilemma using the OpenSpiel
framework.
"""

from simulators.base_simulator import GameSimulator
from utils.llm_utils import generate_prompt, llm_decide_move
from typing import List, Dict, Any
import random

class MatrixGameSimulator(GameSimulator):
    """Simulator for Matrix Games."""

    def simulate(self) -> Dict[str, int]:
        """Simulates a matrix game.

        Returns:
            Dict[str, int]: The scores for each LLM.
        """
        self.scores = {name: 0 for name in self.llms.keys()}  # Reset scores
        state = self.game.new_initial_state()

        while not state.is_terminal():
            self.log_progress(state)  # Log state progress

            # Collect actions for both players
            actions = [
                self._get_action(player, state, state.legal_actions(player))
                for player in range(2)
            ]
            state.apply_actions(actions)

        # Gather final scores
        final_scores = state.returns()
        for i, score in enumerate(final_scores):
            if i < len(self.llms):
                self.scores[list(self.llms.keys())[i]] += score

        self.save_results(state, final_scores)  # Save results to JSON
        return self.scores
