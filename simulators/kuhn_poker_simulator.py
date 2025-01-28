"""Simulator for Kuhn Poker.

This module implements the KuhnPokerSimulator class, which simulates games of
Kuhn Poker using the OpenSpiel framework.

For Kuhn Poker, the game mechanics involve:

- Betting rounds where decisions depend on the game state and potential strategies.
- Chance nodes, which require specific handling (e.g., dealing cards).
"""

from simulators.base_simulator import GameSimulator
from agents.llm_utils import llm_decide_move
from typing import Any
import random


class KuhnPokerSimulator(GameSimulator):
    """Simulator for Kuhn Poker."""

    def _handle_chance_node(self, state: Any):
        """Handle chance nodes for Kuhn Poker.

        Args:
            state (pyspiel.State): The current state of the game.
        """
        outcomes, probabilities = zip(*state.chance_outcomes())
        sampled_outcome = random.choices(outcomes, probabilities)[0]
        state.apply_action(sampled_outcome)

    def _get_action(self, player: int, state: Any, legal_actions: list) -> int:
        """Gets the action for the current player.

        Uses the dedicated Kuhn Poker prompt if the player type is LLM.

        Args:
            player: The index of the current player.
            state: The current game state.
            legal_actions: The legal actions available for the player.

        Returns:
            int: The action selected by the player.
        """
        # If the player type is LLM, use the specialized Kuhn Poker prompt
        if player < len(self.llms):
            model_name = list(self.llms.keys())[player]
            llm = self.llms[model_name]
            prompt = self._generate_poker_prompt(state, legal_actions, player)
            return llm_decide_move(llm, prompt, tuple(legal_actions))  # Convert to tuple

        # For all other cases, defer to the parent class's implementation
        return super()._get_action(player, state, legal_actions)



    def _generate_poker_prompt(self,state: Any, legal_actions: list, player: int) -> str:
        """Generates a detailed prompt for Kuhn Poker using OpenSpiel's state.

        Args:
            state (pyspiel.State): The current game state.
            legal_actions (list): Legal actions available to the player.
            player (int): The index of the current player.

        Returns:
            str: A natural language prompt describing the game state and options.
        """
        # Extract player-specific observation
        observation = state.observation_string(player)

        # Map actions to readable terms
        action_map = {0: "PASS (no additional bet)", 1: "BET (add to the pot)"}
        actions_str = "\n".join(f"{action}: {action_map[action]}" for action in legal_actions)

        # Build the prompt
        prompt = (
            f"You are Player {player + 1} in a game of Kuhn Poker.\n"
            f"Your private observation: {observation}\n"
            f"The goal is to maximize your winnings based on your card and the pot.\n\n"
            f"Available actions:\n{actions_str}\n\n"
            "What action do you choose? Reply with the number corresponding to your action."
        )
        return prompt
