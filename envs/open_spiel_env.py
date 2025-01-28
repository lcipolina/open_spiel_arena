"""
open_spiel_env.py

Implements a Gym-like environment on top of an OpenSpiel game.
"""

from typing import Dict, Any, List
import random
from envs.base_env import BaseEnv, PlayerId


class OpenSpielEnv(BaseEnv):
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

    def step(self, action: int):
        """
        Applies the given action to the environment, then returns (observation, reward, done, info).

        Args:
            action (int): The action to apply. Should be chosen by an external agent.

        Returns:
            observation (Any): Observation after the action.
            reward (float): The reward for this step.
            done (bool): Whether the episode is finished.
            info (dict): Additional diagnostic information (e.g. final scores if done).
        """
        if self.state.is_chance_node():  #TODO (lck look into this)
            # If it's a chance node, handle it automatically.
            # In many OpenSpiel games, chance nodes are built into the state transitions, but
            # if you need to manage them manually, do it here. For now, let's raise an error
             # Potentially you do not apply_action(...) here because chance is random
            # We'll raise an error to remind you to implement it if needed
            print("Chance node detected. REVISE THIS!")
            self._handle_chance_node()
            #raise NotImplementedError("Chance node handling not implemented in step().")

        # Apply the action
        self.state.apply_action(action)

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
        """Cleanup if needed."""
        pass

    # ----------------------------------------------------------------
    # Additional methods
    # ----------------------------------------------------------------

    def _handle_chance_node(self):
        outcomes, probabilities = zip(*self.state.chance_outcomes())
        chosen_outcome = self.random_generator.choices(outcomes, probabilities, k=1)[0]
        self.state.apply_action(chosen_outcome)

    def _collect_actions_simultaneous(self) -> List[int]:
        """Collects actions for all players in a simultaneous-move game."""
        return [
            random.choice(self.state.legal_actions(p))
            for p in range(self.game.num_players())
        ]

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
