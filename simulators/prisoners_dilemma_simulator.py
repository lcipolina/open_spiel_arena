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
        state = self.game.new_initial_state()
        scores = {name: 0 for name in self.llms.keys()}
        iteration = 0

        while not state.is_terminal():
            if iteration >= self.max_iterations:
                print(f"Reached maximum iterations: {self.max_iterations}. Ending simulation.")
                break

            # Handle chance nodes
            if state.is_chance_node():
                print("Chance node encountered. Applying random action.")
                action = state.legal_actions()[0]  # Use the default chance action
                state.apply_action(action)
                continue

            # Get actions for all players
            actions = [
                self._get_action(player, state, state.legal_actions(player))
                for player in range(2)
            ]

            # Apply the actions simultaneously
            state.apply_actions(actions)
            iteration += 1

        # Gather final scores
        final_scores = state.returns()
        for i, score in enumerate(final_scores):
            if i < len(self.llms):
                scores[list(self.llms.keys())[i]] += score

        print(f"Final state of {self.game_name}:\n{state}")
        print(f"Scores: {scores}")
        return scores

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
            return random.choice(legal_actions)
        elif self.play_against_itself:
            model_name = list(self.llms.keys())[player % len(self.llms)]
            llm = self.llms[model_name]
            prompt = generate_prompt(self.game_name, str(state), legal_actions)
            return llm_decide_move(llm, prompt, tuple(legal_actions))  # Convert to tuple
        elif player < len(self.llms):
            model_name = list(self.llms.keys())[player]
            llm = self.llms[model_name]
            prompt = generate_prompt(self.game_name, str(state), legal_actions)
            return llm_decide_move(llm, prompt, tuple(legal_actions))  # Convert to tuple
        return legal_actions[0]  # Default fallback action
