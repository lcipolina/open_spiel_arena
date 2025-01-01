"""Simulator for Kuhn Poker.

This module implements the KuhnPokerSimulator class, which simulates games of
Kuhn Poker using the OpenSpiel framework.

For Kuhn Poker, the game mechanics involve:

- Betting rounds where decisions depend on the game state and potential strategies.
- Chance nodes, which require specific handling (e.g., dealing cards).
"""

from simulators.base_simulator import GameSimulator
from utils.llm_utils import generate_prompt, llm_decide_move
from typing import List, Dict, Any
import random

class KuhnPokerSimulator(GameSimulator):
    """Simulator for Kuhn Poker."""

    def simulate(self, rounds: int = 1) -> Dict[str, Any]:
        """Simulates the game for multiple rounds.

        Args:
            rounds: Number of times the game should be played.

        Returns:
            Dict[str, Any]: Summary of results for all rounds.
        """
        # Inherit base functionality but extend for chance nodes
        outcomes = self._initialize_outcomes()

        for _ in range(rounds):
            # Reset scores for a single round
            self.scores = {name: 0 for name in self.llms.keys()}
            state = self.game.new_initial_state()

            while not state.is_terminal():
                self.log_progress(state)
                current_player = state.current_player()

                if state.is_chance_node():  # Game-specific logic for chance nodes
                    self._handle_chance_node(state)
                    continue

                legal_actions = state.legal_actions(current_player)
                action = self._get_action(current_player, state, legal_actions)
                state.apply_action(action)

            # Record outcomes
            final_scores = state.returns()
            self._record_outcomes(final_scores, outcomes)

        return outcomes

    def _handle_chance_node(self, state: Any) -> None:
        """Handles chance nodes in Kuhn Poker.

        Args:
            state: The current game state.
        """
        print("Chance node encountered. Applying random action.")
        action = random.choice(state.legal_actions())
        state.apply_action(action)

    def _get_action(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Gets the action for the current player.

        Args:
            player: The index of the current player.
            state: The current game state.
            legal_actions: The legal actions available for the player.

        Returns:
            int: The action selected by the player.
        """
        # Check if the player is a random bot
        if self.random_bot and player == 1:
            return random.choice(legal_actions)

        # Handle LLM decision-making
        if player < len(self.llms):
            model_name = list(self.llms.keys())[player]
            llm = self.llms[model_name]
            prompt = self._generate_poker_prompt(state, legal_actions)
            return llm_decide_move(llm, prompt, tuple(legal_actions))  # Convert to tuple

        # Fallback action
        return legal_actions[0]

    def _generate_poker_prompt(self, state: Any, legal_actions: List[int]) -> str:
        """Generates a specialized prompt for Kuhn Poker.

        Args:
            state: The current game state.
            legal_actions: The legal actions available for the player.

        Returns:
            str: A natural language prompt describing the game state and legal actions.
        """
        return (
            f"Kuhn Poker Game:\n"
            f"Current state: {state}\n"
            f"Your options: {legal_actions}\n"
            "What action do you choose?"
        )
