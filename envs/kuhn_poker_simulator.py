"""This module implements the KuhnPokerSimulator class, which simulates games of
Kuhn Poker using the OpenSpiel framework.

For Kuhn Poker, the game mechanics involve:

- Betting rounds where decisions depend on the game state and potential strategies.
- Chance nodes, which require specific handling (e.g., dealing cards).
"""

import random
from typing import Any, Dict
from envs.open_spiel_env import OpenSpielEnv


class KuhnPokerSimulator(OpenSpielEnv):
    """Simulator for Kuhn Poker."""

    def __init__(self, game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None):
        """
        Args:
            game: The OpenSpiel game object.
            game_name: A string representing the name of the game.
            player_types: A dictionary mapping player IDs to their types (e.g., human, random).
            max_game_rounds: Maximum number of rounds
                             for iterated games (optional, default is None).
        """
        super().__init__(game, game_name, player_types, max_game_rounds)
        self.current_player = 0  # Placeholder for the current player index


    def _state_to_observation(self) -> Dict[str, Any]:
        """
        Generate the observation for the matrix game.

        Returns:
            Dict[str, Any]: Observation dictionary containing:
                - state_string: A placeholder for state description (None in RPS).
                - legal_actions: A list of valid actions for each player.
                - info: A string providing action descriptions.

        # Observation tensor encodes: [player0, player1, J,Q,K, pot0,pot1]
        # Current player (one-hot) (ex:[1,0] if it's player 1).
        # Current card (one-hot) (ex:[1,0,0] for [J,Q,K]).
        # Initial pot contribution (ex: [1,1]).
        Example Output (observation_tensor() as a List)

        """

        # Ensure chance nodes are handled before extracting observations
        while self.state.is_chance_node():
            outcomes, probs = zip(*self.state.chance_outcomes())  # distibution over outcomes as a list of (outcome, probability) pairs
            action = random.choices(outcomes, probs)[0]  # Pick a random outcome and convert from list to scalar.
            self.state.apply_action(action)

        # Set the current player (first to act)
        self.current_player = self.state.current_player()

        # Private data for the current player
        tensor_observation = self.state.observation_tensor(self.current_player) # One-hot encoded tensor
        legal_actions = self.state.legal_actions(self.current_player)
        prompt_observation = self._generate_prompt(self.state, legal_actions)

        return {
            "state": tensor_observation,  # RL agent used this
            "legal_actions":legal_actions,
            "info": None,
            "prompt": prompt_observation
        }


    def _generate_prompt(self, state: Any, legal_actions: list) -> dict:
        """Generates a detailed observation for Kuhn Poker.

        Args:
            state (pyspiel.State): The current game state.
            legal_actions (list): Legal actions available to the player.

        Returns:
            dict: A structured observation containing:
                - "tensor": One-hot encoded observation for RL policies.
                - "prompt": A human-readable prompt for LLMs & humans.
        """
        # RL Observation: One-hot encoded tensor
        tensor_observation = state.observation_tensor(self.current_player)

        # Extract private card from tensor
        private_card = self.extract_private_card_from_tensor(tensor_observation)

        # Extract structured betting history
        betting_history = self._get_betting_history(state)

        # Extract total pot size and player's contribution
        total_pot = sum(tensor_observation[-2:])  # Last two values are pot contributions
        player_contribution = tensor_observation[-2 + self.current_player]  # Index -2 (P1) or -1 (P2)

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

        # Build the natural language prompt
        prompt = (
            f"You are Player {self.current_player} in the game Kuhn Poker.\n"
            f"Your private card: {private_card}\n"
            f"Betting history: {betting_history}\n"
            f"Total pot size: {total_pot} chips\n"
            f"Your contribution: {player_contribution} chips\n\n"
            f"Available actions:\n{actions_str}\n\n"
            "What action do you choose? Reply with the number corresponding to your action."
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



    