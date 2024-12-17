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

            # Collect actions for all players
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

    def _get_action(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Gets the action for the current player.

        Args:
            player: The index of the current player.
            state: The current game state.
            legal_actions: The legal actions available for the player.

        Returns:
            int: The action selected by the player.
        """
        if self.random_bot and player == 1:
            # Player 1 acts as a random bot
            return random.choice(legal_actions)
        elif self.play_against_itself:
            # Both players are controlled by LLMs (self-play)
            model_name = list(self.llms.keys())[player % len(self.llms)]
            llm = self.llms[model_name]
            prompt = generate_prompt(self.game_name, str(state), legal_actions)
            return llm_decide_move(llm, prompt, tuple(legal_actions))  # Convert to tuple
        elif player < len(self.llms):
            # Player is controlled by an LLM
            model_name = list(self.llms.keys())[player]
            llm = self.llms[model_name]
            prompt = generate_prompt(self.game_name, str(state), legal_actions)
            return llm_decide_move(llm, prompt, tuple(legal_actions))  # Convert to tuple
        return legal_actions[0]  # Default fallback action
