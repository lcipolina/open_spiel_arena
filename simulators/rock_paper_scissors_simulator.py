# simulators/rock_paper_scissors_simulator.py
"""Simulator for Rock-Paper-Scissors.

This module implements the RockPaperScissorsSimulator class, which simulates games of
Rock-Paper-Scissors using the OpenSpiel framework.
"""

import random
from .base_simulator import GameSimulator
from typing import List, Dict, Any

class RockPaperScissorsSimulator(GameSimulator):
    """Simulator for Rock-Paper-Scissors."""

    def simulate(self) -> Dict[str, int]:
        """Simulates a game of Rock-Paper-Scissors.

        Returns:
            Dict[str, int]: The scores for each LLM.
        """
        state = self.game.new_initial_state()
        scores = {name: 0 for name in self.llms.keys()}

        while not state.is_terminal():
            actions = [
                self._get_action(player, state, state.legal_actions(player))
                for player in range(2)
            ]
            state.apply_actions(actions)

        final_scores = state.returns()
        for i, score in enumerate(final_scores):
            if i < len(self.llms):
                scores[list(self.llms.keys())[i]] += score

        print(f"Final state of {self.game_name}:\n{state}")
        print(f"Scores: {scores}")
        return scores




