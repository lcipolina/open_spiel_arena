"""Simulator for Matching Pennies (3-player).

This module implements the MatchingPenniesSimulator class, which simulates games of
Matching Pennies with three players using the OpenSpiel framework.
"""

from simulators.base_simulator import GameSimulator
from utils.llm_utils import generate_prompt, llm_decide_move
from typing import List, Dict, Any
import random

class MatchingPenniesSimulator(GameSimulator):
    """Simulator for Matching Pennies (3-player)."""

    def simulate(self) -> Dict[str, int]:
        """Simulates a game of Matching Pennies with three players.

        Returns:
            Dict[str, int]: The scores for each LLM.
        """
        self.scores = {name: 0 for name in self.llms.keys()}  # Reset scores
        state = self.game.new_initial_state()

        while not state.is_terminal():
            self.log_progress(state)  # Log state progress

            # Collect actions for all players
            actions = [
                self._get_action(player, state, state.legal_actions(player))
                for player in range(3)  # Adjusted for 3 players
            ]
            state.apply_actions(actions)

        # Gather final scores
        final_scores = state.returns()
        for i, score in enumerate(final_scores):
            if i < len(self.llms):
                self.scores[list(self.llms.keys())[i]] += score

        self.save_results(state, final_scores)  # Save results to JSON
        return self.scores
