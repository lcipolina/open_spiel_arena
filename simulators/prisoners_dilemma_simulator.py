# simulators/prisoners_dilemma_simulator.py
"""Simulator for Iterated Prisoner's Dilemma.

This module implements the PrisonersDilemmaSimulator class, which simulates games of
the Iterated Prisoner's Dilemma using the OpenSpiel framework.
"""

import random
from simulators.base_simulator import GameSimulator
from utils.llm_utils import generate_prompt, llm_decide_move
from typing import List, Dict, Any

class PrisonersDilemmaSimulator(GameSimulator):
    """Simulator for Iterated Prisoner's Dilemma."""

    def __init__(self, game: Any, game_name: str, llms: Dict[str, Any],
                 random_bot: bool = False, play_against_itself: bool = False,
                 max_iterations: int = 50) -> None:
        super().__init__(game, game_name, llms, random_bot, play_against_itself)
        self.max_iterations = max_iterations

    def simulate(self) -> Dict[str, int]:
        """Simulates a game of Iterated Prisoner's Dilemma.

        Returns:
            Dict[str, int]: The scores for each LLM.
        """
        self.scores = {name: 0 for name in self.llms.keys()}  # Reset scores
        state = self.game.new_initial_state()
        iteration = 0

        while not state.is_terminal():
            if iteration >= self.max_iterations:
                print(f"Reached maximum iterations: {self.max_iterations}. Ending simulation.")
                break

            # Handle chance nodes
            if state.is_chance_node():
                print("Chance node encountered. Applying random action.")
                action = state.legal_actions()[0]
                state.apply_action(action)
                continue

            # Collect actions for both players (from base class)
            actions = [
                self._get_action(player, state, state.legal_actions(player))
                for player in range(2)
            ]

            # Apply actions simultaneously
            state.apply_actions(actions)
            iteration += 1

        # Gather final scores
        final_scores = state.returns()
        for i, score in enumerate(final_scores):
            if i < len(self.llms):
                self.scores[list(self.llms.keys())[i]] += score

        self.save_results(state, final_scores)  # Save results
        return self.scores
