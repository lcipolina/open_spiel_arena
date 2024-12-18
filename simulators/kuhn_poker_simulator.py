"""Simulator for Kuhn Poker.

This module implements the KuhnPokerSimulator class, which simulates games of
Kuhn Poker using the OpenSpiel framework.
"""

from simulators.base_simulator import GameSimulator
from utils.llm_utils import generate_prompt, llm_decide_move
from typing import List, Dict, Any
import random

class KuhnPokerSimulator(GameSimulator):
    """Simulator for Kuhn Poker."""

    def simulate(self) -> Dict[str, int]:
        """Simulates a game of Kuhn Poker.

        Returns:
            Dict[str, int]: The scores for each LLM.
        """
        self.scores = {name: 0 for name in self.llms.keys()}  # Reset scores
        state = self.game.new_initial_state()

        while not state.is_terminal():
            self.log_progress(state)  # Log the state progress
            current_player = state.current_player()
            legal_actions = state.legal_actions(current_player)

            # Get action for the current player
            action = self._get_action(current_player, state, legal_actions)
            state.apply_action(action)

        # Gather final scores
        final_scores = state.returns()
        for i, score in enumerate(final_scores):
            if i < len(self.llms):
                self.scores[list(self.llms.keys())[i]] += score

        self.save_results(state, final_scores)  # Save results to JSON
        return self.scores
