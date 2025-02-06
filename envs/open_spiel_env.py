"""
open_spiel_env.py

Implements a Gymnasium-like environment on top of an OpenSpiel game.
"""

from typing import Any, Dict, List, Tuple, Union, Optional
import random
from abc import ABC


class OpenSpielEnv(ABC):
    """Environment for OpenSpiel.

    Handles common functionality like state transitions, outcomes recording,
    and logging.
    """

    def __init__(self,
                 game: Any,
                 game_name: str,
                 player_types: Dict[str, str],
                 max_game_rounds: int = None):
        """
        Args:
            game (Any): The OpenSpiel game object being simulated.
            game_name (str): A human-readable name for the game.
            player_type (Dict[str, str]): Maps "Player 1", "Player 2", ... to their types (human, random, llm, etc.).
            max_game_rounds (int): Maximum number of rounds for iterated games. Ignored by single-shot games.
        """
        self.game = game
        self.game_name = game_name
        self.player_types = player_types # List of strings
        self.max_game_rounds = max_game_rounds  # For iterated games
        self.state = None
        self.info = {}
        self.terminated, self.truncated = False, False

    def reset(self, seed: Optional[int]=None) -> Tuple[str, Dict[str, Any]]:
        """
        Resets the environment to an initial state and returns an initial observation.

        Returns:
            Tuple[str, Dict[str, Any]]:
                - str: A string representation of the initial state.
                - Dict[str, Any]: Additional info
        """
        if seed is not None:
            self.seed(seed)

        self.state = self.game.new_initial_state() # Instantiates a pyspiel game
        self.terminated = False
        self.truncated = False
        self.info = {}

        # Handle chance nodes first (e.g., dealing cards in Kuhn Poker)
        if self.state.is_chance_node():
            self._solve_chance_nodes()
            return self._state_to_observation(), self.info

        return self._state_to_observation(), self.info

    def apply_action(self, action: int):
        """Applies the given action to the environment.

        Args:
            action int: If the game is turn-based, it is an integer.
        """
        self.state.apply_action(action)

    def step(self, action_dict: Dict[int, int]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Applies the given action(s) to the environment and returns the new state.

        Args:
            action_dict (Dict[int, int]): A dictionary mapping agent IDs to actions.
                - For turn-based games: {current_player: action}
                - For simultaneous games: {player_0: action_0, player_1: action_1, ...}

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: A tuple containing:
                - observation (Any): The resulting state after the action.
                - reward (float): The reward obtained from this step.
                - terminated (bool): Whether the episode has ended normally.
                - truncated (bool): Whether the episode ended due to `max_game_rounds`.
                - info (Dict[str, Any]): Additional diagnostic information (e.g., final scores if done).
        """

        # Handle chance nodes
        if self.state.is_chance_node():
            self._solve_chance_nodes()
            return self._state_to_observation(), {}, False, False, {}

        # Handle simultaneous move games
        if self.state.is_simultaneous_node():
            actions = [action_dict[player] for player in sorted(action_dict.keys())]
            self.state.apply_actions(actions)  # Multi-agent moves
        else:
            current_player = list(action_dict.keys())[0]
            self.state.apply_action(action_dict[current_player]) # Single action

        # Stepwise reward for each OpenSpiel-indexed agent
        reward_dict = self._compute_reward()

        # Check termination due to game end
        self.terminated = self.state.is_terminal()

        # Check truncation due to max rounds (condition for iterated games)
        self.truncated = (
            self.max_game_rounds is not None
             and self.state.move_number() >= self.max_game_rounds
        )

        # If the game is finished, store final scores; otherwise, update current player
        if self.terminated or self.truncated:
            self.info["final_scores"] = self.state.returns()
            observation_dict = {agentID: None for agentID in list(action_dict.keys())} # No observation when the game ends
        else:
            observation_dict = self._state_to_observation() # Get next observation for all agents

        return observation_dict, reward_dict, self.terminated, self.truncated, self.info

    def render(self, mode: str = 'human'):
        """Print out the current state of the game."""
        if mode == 'human':
            print(f"Current state of {self.game_name}:\n{self.state}")

    def seed(self, seed: int = None):
        """
        Sets the random seed for the environment.

        Args:
            seed (int): The random seed.
        """
        self.random_generator = random.Random(seed)
        self.state.set_seed(seed)

    def close(self):
        """Cleanup."""
        pass

    # ----------------------------------------------------------------
    # Additional methods
    # ----------------------------------------------------------------

    def _state_to_observation(self) -> Dict[int, Dict[str, Any]]:
        """Returns the observation for each agent in the game.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping from agent ID to their respective observations.
        """
        return {
            agent_id: {
                "state_string": self.state.observation_string(agent_id),
                "legal_actions": self.state.legal_actions(agent_id),
                "prompt": None  # Can be overridden in child classes
            }
            for agent_id in range(self.state.num_players())  # Generate for ALL players
        }


    def _state_to_observation_old(self, action_dict: Dict[int, int]) -> Dict[int, Dict[str, Any]]:
        """Returns the observation for each agent in the game.

        Args:
        action_dict (Dict[int, int]): Mapping of agent ID to their last action.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping from agent ID to their respective observations.
        """
        observation_dictionary = {
        agent_id: {
            "state_string": self.state.observation_string(agent_id),
            "legal_actions": self.state.legal_actions(agent_id),
            "prompt": None  # Can be overridden in child classes
        }
        for agent_id in action_dict.keys()
        }
        return observation_dictionary


    def _solve_chance_nodes(self) -> None:
        """Automatically plays chance nodes by selecting outcomes based on probabilities.

        Many OpenSpiel games involve chance nodes (e.g., dealing cards).
        This method ensures that chance nodes are resolved before player actions.
        """
        while self.state.is_chance_node():
            outcomes, probs = zip(*self.state.chance_outcomes())  # List of (outcome, probability)
            action = random.choices(outcomes, probs)[0]  # Pick a random outcome
            self.state.apply_action(action)  # Apply the chosen chance action


    def _compute_reward(self) -> Dict[int, float]:
        """Returns rewards indexed by OpenSpiel player indices (0, 1, ...)."""
        return {player: self.state.player_reward(player) for player in range(self.state.num_players())}
