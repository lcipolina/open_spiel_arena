"""
open_spiel_env.py

Implements a Gymnasium-like environment on top of an OpenSpiel game.
"""

from typing import Any, Dict, List, Tuple, Union
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
        self.player_types = player_types
        self.max_game_rounds = max_game_rounds  # For iterated games
        self.state = None
        self.rewards = {}

    def reset(self) -> str:
        """
        Resets the environment to an initial state and returns an initial observation.

        Returns:
            str: A string representation of the initial state (or any other observation format).
        """
        self.state = self.game.new_initial_state() # Instantiates a pyspiel game
        self.rewards = {name: 0 for name in self.player_types}
        return self._state_to_observation()

    def apply_action(self, action: int):
        """Applies the given action to the environment.

        Args:
            action int: If the game is turn-based, it is an integer.
        """
        self.state.apply_action(action)

    def step(self, action: Union[int, List[int]]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Applies the given action(s) to the environment and returns the new state.

        Args:
            action (Union[int, List[int]]): The action to apply. If the game is
                turn-based, it is an integer. If the game is simultaneous-move,
                it is a list of actions (one for each player).

        Returns:
            Tuple[Any, float, bool, Dict[str, Any]]: A tuple containing:
                - observation (Any): The resulting state or observation after the action.
                - reward (float): The reward obtained from this step.
                - done (bool): Whether the episode has ended.
                - info (Dict[str, Any]): Additional diagnostic information (e.g., final scores if done).
        """

        # Apply the action
        self.apply_action(action)

        # Stepwise reward for each agent
        reward_dict = self._compute_reward()

        # Check termination
        done = self.state.is_terminal()
        if (self.max_game_rounds is not None
                and self.state.move_number() >= self.max_game_rounds
                ):   # Condition for iterated games
            done = True

        # Build the new observation
        observation = self._state_to_observation() if not done else None

        # Accumulated rewards for all players
        info = (
            {"final_scores": self.state.returns()}
            if done
            else {}
        )

        return observation, reward_dict, done, info

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

    # TODO: these forst two methods might need to be deleted
    def _handle_chance_node(self):
        outcomes, probabilities = zip(*self.state.chance_outcomes())
        chosen_outcome = self.random_generator.choices(outcomes, probabilities, k=1)[0]
        self.state.apply_action(chosen_outcome)


    def _state_to_observation(self) -> Dict[str, Any]:
        return {
            "state_string":  self.state.observation_string(),
            "legal_actions": self.state.legal_actions(),
        }

    def _compute_reward(self) -> Dict[int, float]:
        """
        Compute the step rewards for all agents at the current step.

        Returns:
            Dict[int, float]: A dictionary mapping agent IDs to their step rewards.
        """
        players_list = range(self.state.num_players())
        rewards = {
            player: self.state.player_reward(player) for player in players_list
        }
        return rewards
