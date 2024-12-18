# simulators/rock_paper_scissors_simulator.py
"""Simulator for Rock-Paper-Scissors.

This module implements the RockPaperScissorsSimulator class, which simulates games of
Rock-Paper-Scissors using the OpenSpiel framework.
"""

import random
from simulators.base_simulator import GameSimulator
from utils.llm_utils import generate_prompt, llm_decide_move
from typing import List, Dict, Any

class RockPaperScissorsSimulator(GameSimulator):
    """Simulator for Rock-Paper-Scissors."""

    def simulate(self) -> Dict[str, int]:
        """Simulates a game of Rock-Paper-Scissors.

        Returns:
            Dict[str, int]: The scores for each LLM.
        """
        self.scores = {name: 0 for name in self.llms.keys()}  # Reset scores
        state = self.game.new_initial_state()

        while not state.is_terminal():
            self.log_progress(state)  # Use the base class logging

            # Collect actions for all players (use base class)
            actions = [
                self._get_action(player, state, state.legal_actions(player))
                for player in range(2)
            ]

            # Apply actions simultaneously
            state.apply_actions(actions)

        # Gather final scores
        final_scores = state.returns()
        for i, score in enumerate(final_scores):
            if i < len(self.llms):
                self.scores[list(self.llms.keys())[i]] += score

        # Save results and return scores
        self.save_results(state, final_scores)
        return self.scores
