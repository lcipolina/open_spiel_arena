"""This module implements the KuhnPokerSimulator class, which simulates games of
Kuhn Poker using the OpenSpiel framework.

For Kuhn Poker, the game mechanics involve:

- Betting rounds where decisions depend on the game state and potential strategies.
- Chance nodes, which require specific handling (e.g., dealing cards).
"""

from envs.open_spiel_env import OpenSpielEnv
from agents.llm_utils import llm_decide_move
from typing import Any, List, Dict
import random


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


    def _state_to_observation(self) -> Dict[str, Any]:
        """
        Generate the observation for the matrix game.

        Returns:
            Dict[str, Any]: Observation dictionary containing:
                - state_string: A placeholder for state description (None in RPS).
                - legal_actions: A list of valid actions for each player.
                - info: A string providing action descriptions.

              # Observation tensor encodes:
        # Current player (ex:[1,0] if it's player 1).
        # Current card (ex:[1,0,0] for [J,Q,K]).
        # Pot contribution (ex: [2,2]).
        Example Output (observation_tensor() as a List)

        """
        while self.state.is_chance_node():
            outcomes, probs = zip(*self.state.chance_outcomes())  # distibution over outcomes as a list of (outcome, probability) pairs
            action = random.choices(outcomes, probs)[0]  # Pick a random outcome and convert from list to scalar.
            self.state.apply_action(action)

        # Set the current player (first to act)
        self.current_player = self.state.current_player()

        # Private observation for the current player
        observation = self.state.observation_tensor(self.current_player)
        valid_actions = self.state.legal_actions(self.current_player)
        action_description = 'a'

        return {
            "state": None,  # No meaningful observation in simultaneous games
            "legal_actions": self.state.legal_actions(self.current_player),
            "info": f"Actions available: {action_description}"
        }

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



    def _generate_poker_prompt_old(self,state: Any, legal_actions: list, player: int) -> str:
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

    def _generate_prompt_new_but_old(self, state: Any, legal_actions: list, player: int) -> dict:
        """Generates a detailed observation for Kuhn Poker.

        Args:
            state (pyspiel.State): The current game state.
            legal_actions (list): Legal actions available to the player.
            player (int): The index of the current player.

        Returns:
            dict: A structured observation containing:
                - "tensor": One-hot encoded observation for RL policies.
                - "prompt": A human-readable prompt for LLMs & humans.
        """
        # RL Observation: One-hot encoded tensor
        tensor_observation = state.observation_tensor(player)

        # Human/LLM Observation: Natural Language Prompt
        observation_str = state.observation_string(player)  # Private card & history

        # Extract relevant game information
        history = state.history()  # List of all actions taken
        pot_size = state.pot() if hasattr(state, "pot") else "Unknown"  # OpenSpiel may not expose directly
        last_action = history[-1] if history else "No actions taken yet"

        # Map actions to readable terms
        action_map = {0: "PASS (no additional bet)", 1: "BET (add to the pot)"}
        actions_str = "\n".join(f"{action}: {action_map[action]}" for action in legal_actions)

        # Build the natural language prompt
        prompt = (
            f"You are Player {player + 1} in a game of Kuhn Poker.\n"
            f"Your private card: {observation_str[0]}\n"  # First character is the card (J, Q, K)
            f"Betting history: {' '.join(map(str, history)) if history else 'No actions yet'}\n"
            f"Current pot size: {pot_size}\n"
            f"Last action taken: {last_action}\n\n"
            f"Available actions:\n{actions_str}\n\n"
            "What action do you choose? Reply with only the number corresponding to your action. Do not add any additional text."
        )

        return {
            "tensor": tensor_observation,  # RL policy input
            "prompt": prompt  # LLMs & Humans
        }


#TODO: check if the valid actions really come as [0,1,2,3]??

def _get_betting_history(state: Any) -> str:
    """Extracts a readable betting history from OpenSpiel's game state.

    Args:
        state (pyspiel.State): The current game state.

    Returns:
        str: A formatted betting history with player actions.
    """
    action_map = {0: "Check", 1: "Bet", 2: "Call", 3: "Fold"}

    history = []
    num_players = 2  # Kuhn Poker always has 2 players

    for i, action in enumerate(state.history()):
        player = i % num_players  # Alternates between 0 and 1
        history.append(f"Player {player + 1}: {action_map[action]}")

    return " -> ".join(history) if history else "No actions yet"


def _generate_poker_prompt(self, state: Any, legal_actions: list, player: int) -> dict:
    """Generates a detailed observation for Kuhn Poker.

    Args:
        state (pyspiel.State): The current game state.
        legal_actions (list): Legal actions available to the player.
        player (int): The index of the current player.

    Returns:
        dict: A structured observation containing:
            - "tensor": One-hot encoded observation for RL policies.
            - "prompt": A human-readable prompt for LLMs & humans.
    """
    # RL Observation: One-hot encoded tensor
    tensor_observation = state.observation_tensor(player)

    # Human/LLM Observation: Natural Language Prompt
    observation_str = state.observation_string(player)  # Private card & history

    # Extract structured betting history
    betting_history = _get_betting_history(state)

    # Map actions to readable terms
    action_labels = {
        0: "Check (stay in the game without betting)",
        1: "Bet (add a chip to the pot)",
        2: "Call (match the opponent's bet)",
        3: "Fold (give up and lose the pot)"
    }
    actions_str = "\n".join(f"{action}: {action_labels[action]}" for action in legal_actions)

    # Build the natural language prompt
    prompt = (
        f"You are Player {player + 1} in a game of Kuhn Poker.\n"
        f"Your private card: {observation_str[0]}\n"
        f"Betting history: {betting_history}\n\n"
        f"Available actions:\n{actions_str}\n\n"
        "What action do you choose? Reply with the number corresponding to your action."
    )

    return {
        "tensor": tensor_observation,  # RL policy input
        "prompt": prompt  # LLMs & Humans
    }
