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
        state = self.game.new_initial_state()

        while not state.is_terminal():
            self.log_progress(state)  # Use the base class logging
            current_player = state.current_player()
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

    def _get_action(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Gets the action for the current player.

        Args:
            player: The index of the current player.
            state: The current game state.
            legal_actions: The legal actions for the player.

        Returns:
            int: The action selected by the player.
        """
        if self.random_bot and player == 1:
            return random.choice(legal_actions)
        elif self.play_against_itself:
            return self._llm_decide(player, state, legal_actions)
        elif player < len(self.llms):
            return self._llm_decide(player, state, legal_actions)
        return legal_actions[0]

    # TODO: this seems to be redundant, as it is imported!!! - check
    def _llm_decide(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Uses an LLM to decide the next move.

        Args:
            player: The index of the player.
            state: The current game state.
            legal_actions: The legal actions available.

        Returns:
            int: The action selected by the LLM.
        """
        model_name = list(self.llms.keys())[player]
        llm = self.llms[model_name]
        prompt = generate_prompt(self.game_name, str(state), legal_actions)
        return llm_decide_move(llm, prompt, tuple(legal_actions))
