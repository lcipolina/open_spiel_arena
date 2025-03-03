"""This module implements the KuhnPokerSimulator class, which simulates games of
Kuhn Poker using the OpenSpiel framework.

For Kuhn Poker, the game mechanics involve:

- Betting rounds where decisions depend on the game state and potential strategies.
- Chance nodes, which require specific handling (e.g., dealing cards).
"""

from typing import Any, Dict, Optional
from envs.open_spiel_env import OpenSpielEnv


class KuhnPokerSimulator(OpenSpielEnv):
    """Simulator for Kuhn Poker."""

    def __init__(self, game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None,
                 seed: Optional[int] = None):
        """
        Args:
            game: The OpenSpiel game object.
            game_name: A string representing the name of the game.
            player_types: A dictionary mapping player IDs to their types (e.g., human, random).
            max_game_rounds: Maximum number of rounds
                             for iterated games (optional, default is None).
        """
        super().__init__(game, game_name, player_types, max_game_rounds, seed)


    def _state_to_observation(self) -> Dict[int, Dict[str, Any]]:
        """Returns the observation for each agent in the game.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping from agent ID to their respective observations.
        """
        return {
            agent_id: {
                "state_string": self.state.observation_string(agent_id),
                "legal_actions": self.state.legal_actions(agent_id),
                "prompt": self._generate_prompt(self.state, self.state.legal_actions(agent_id), agent_id)
            }
            for agent_id in range(self.state.num_players())  # Generate for ALL players
        }

    def _generate_prompt(self, state: Any, legal_actions: list, agent_id: int) -> str:
        """Generates a detailed observation for Kuhn Poker.

        Args:
            state (pyspiel.State): The current game state.
            legal_actions (list): Legal actions available to the player.
            agent_id (int): The agent/player ID.

        Returns:
            str: A structured LLM prompt for decision-making.
        """

        if self.state.is_chance_node():  # If in a chance node, return empty observation
            return {}

        # RL Observation: One-hot encoded tensor
        tensor_observation = state.observation_tensor(agent_id)

        # Extract private card from tensor
        private_card = self.extract_private_card_from_tensor(tensor_observation)

        # Extract structured betting history
        betting_history = self._get_betting_history(state)

        # Extract total pot size and player's contribution
        total_pot = sum(tensor_observation[-2:])  # Last two values are pot contributions
        player_contribution = tensor_observation[-2 +  agent_id]  # Index -2 (P1) or -1 (P2)

        # Detect if an opponent has already bet
        previous_actions = state.history()
        opponent_has_bet = 1 in previous_actions  # True if opponent bet

        # Map actions correctly based on game state
        if opponent_has_bet:
            action_labels = {
                0: "Fold (give up and lose the pot)",
                1: "Call (match the opponent's bet)"
            }
        else:
            action_labels = {
                0: "Check (stay in the game without betting)",
                1: "Bet (add a chip to the pot)"
            }

        actions_str = "\n".join(f"{action}: {action_labels[action]}" for action in legal_actions)

        prompt = (
            f"You are Player {agent_id} in the game Kuhn Poker.\n"
            f"Your private card: {private_card}\n"
            f"Betting history: {betting_history}\n"
            f"Total pot size: {total_pot} chips\n"
            f"Your contribution: {player_contribution} chips\n\n"
            f"Available actions:\n{actions_str}\n\n"
            "What action do you choose? Reply **only** with '0' or '1'. Do not repeat the options.."
     )

        return prompt

    def extract_private_card_from_tensor(self,observation_tensor: list) -> str:
        """Extracts the player's private card from the one-hot encoded tensor.

        Args:
            observation_tensor (list): The player's observation tensor.

        Returns:
            str: The player's private card ('J', 'Q', or 'K').
        """
        card_map = {0: "J", 1: "Q", 2: "K"}
        card_index = observation_tensor[2:5].index(1.0)  # Find which card is 1.0
        return card_map.get(card_index, "Unknown")

    def _get_betting_history(self, state: Any) -> str:
        """Extracts a readable betting history from OpenSpiel's game state.

           This function converts the sequence of past actions into a readable format,
           indicating which player took each action. It alternates between Player 1 and
            Player 2 based on turn order.

        Args:
            state (pyspiel.State): The current game state.

        Returns:
            str: A formatted betting history with player actions.
        """

        action_map = {0: "Check", 1: "Bet"}
        betting_history = []
        num_players = 2  # Kuhn Poker always has 2 players

        history = state.history()

        # FIX: Ignore the first `num_players` actions (they are card assignments) # TODO: check with Marc
        betting_actions = history[num_players:]

        # FIX: If no betting has happened yet, return "No actions yet" #TODO: check with MArc
        if len(betting_actions) == 0:
            return "No actions yet"

        # Iterate over the betting actions
        for i, action in enumerate(betting_actions):
            player = i % num_players  # Alternates between Player 1 (0) and Player 2 (1)

            # Adjust the action name depending on the betting round
            if 1 in betting_actions[:i]:  # If a bet was made before this action
                action_label = "Call" if action == 1 else "Fold"
            else:
                action_label = action_map.get(action, "Unknown")

            betting_history.append(f"Player {player + 1}: {action_label}")

        return " -> ".join(betting_history)
