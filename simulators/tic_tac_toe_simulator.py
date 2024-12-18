# simulators/tic_tac_toe_simulator.py
"""Simulator for Tic-Tac-Toe.

This module implements the TicTacToeSimulator class, which simulates games of
Tic-Tac-Toe using the OpenSpiel framework.
"""

import random
from simulators.base_simulator import GameSimulator
from utils.llm_utils import generate_prompt, llm_decide_move
from typing import List, Dict, Any

class TicTacToeSimulator(GameSimulator):
    """Simulator for Tic-Tac-Toe."""

    def simulate(self) -> Dict[str, int]:
        """Simulates a game of Tic-Tac-Toe.

        Returns:
            Dict[str, int]: The scores for each LLM.
        """
        self.scores = {name: 0 for name in self.llms.keys()}  # Reset scores
        state = self.game.new_initial_state()

        while not state.is_terminal():
            self.log_progress(state)  # Use the base class logging
            current_player = state.current_player()

            # Handle invalid player states
            if current_player < 0:
                self._apply_default_action(state)
                continue

            legal_actions = state.legal_actions(current_player)
            action = self._get_action(current_player, state, legal_actions)
            state.apply_action(action)

        final_scores = state.returns()
        for i, score in enumerate(final_scores):
            if i < len(self.llms):
                self.scores[list(self.llms.keys())[i]] += score

        self.save_results(state, final_scores)  # Save results
        return self.scores
