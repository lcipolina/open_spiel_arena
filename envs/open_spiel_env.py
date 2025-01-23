"""
open_spiel_env.py

Implements a Gym-like environment on top of an OpenSpiel game.
"""

from typing import Dict, Any, List
import random
import os
import json

from envs.base_env import BaseEnv, PlayerId


class OpenSpielEnv(BaseEnv):
    """Environment for OpenSpiel.

    Handles common functionality like state transitions, outcomes recording,
    and logging.
    """

    def __init__(self,
                 game: Any,
                 game_name: str,
                 player_type: Dict[str, str],
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
        self.player_type = player_type
        self.max_game_rounds = max_game_rounds  # For iterated games
        self.state = None
        self.scores = {}  # Scoreboard, e.g., { "Player 1": 0, "Player 2": 0, ... }

    def reset(self) -> str:
        """
        Resets the environment to an initial state and returns an initial observation.

        Returns:
            str: A string representation of the initial state (or any other observation format).
        """
        self.state = self.game.new_initial_state()
        self.scores = {name: 0 for name in self.player_type.keys()}
        return self._state_to_observation(self.state)

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
        if self._is_chance_node():  #TODO (lck look into this)
            # If it's a chance node, handle it automatically.
            # In many OpenSpiel games, chance nodes are built into the state transitions, but
            # if you need to manage them manually, do it here. For now, let's raise an error
             # Potentially you do not apply_action(...) here because chance is random
            # We'll raise an error to remind you to implement it if needed
            raise NotImplementedError("Chance node handling not implemented in step().")

        # Apply the action
        self.state.apply_action(action)

        # Build the new observation
        observation = self._state_to_observation(self.state)

        # Compute reward
        reward = self._compute_reward(self.state)

        # Check termination
        done = self.state.is_terminal()
        if (self.max_game_rounds is not None
                and self.state.move_number() >= self.max_game_rounds):
            done = True

        # info dict for debugging or final scores
        info = {}
        if done:
            final_scores = self.state.returns()  # returns an array
            info['final_scores'] = final_scores

        return observation, reward, done, info

    def render(self, mode: str = 'human'):
        """Print out the current state of the game."""
        if mode == 'human':
            print(f"Current state of {self.game_name}:\n{self.state}")

    def close(self):
        """Cleanup if needed."""
        pass

    # ----------------------------------------------------------------
    # Additional methods
    # ----------------------------------------------------------------

    def _is_chance_node(self) -> bool:
        """Check if the current player is CHANCE."""
        current_player = self.state.current_player()
        player_id = self.normalize_player_id(current_player)
        return (player_id == PlayerId.CHANCE.value)

    def _handle_chance_node(self):
        """Handle chance nodes. Default behavior raises an error."""
        raise NotImplementedError("Chance node handling not implemented for this game.")

    def _collect_actions_simultaneous(self) -> List[int]:
        """Collects actions for all players in a simultaneous-move game."""
        return [
            random.choice(self.state.legal_actions(p))
            for p in range(self.game.num_players())
        ]

    def _state_to_observation(self, state: Any) -> str:
        """
        Convert the current OpenSpiel state into an observation.
        You can return any format (string, dict, custom object).
        """
        return str(state)

    def _compute_reward(self, state: Any) -> float:
        """
        Compute the reward at the current step. Many environments return
        0 reward until the game ends, then final outcome as reward.
        """
        if not state.is_terminal():
            return 0.0
        # Example: sum of final scores
        return sum(state.returns())

    def _initialize_outcomes(self) -> Dict[str, Any]:
        """Initializes an outcomes dictionary to track wins, losses, ties, etc."""
        return {
            "wins": {name: 0 for name in self.player_type.keys()},
            "losses": {name: 0 for name in self.player_type.keys()},
            "ties": 0
        }

    def record_outcomes(self, final_scores: List[float], outcomes: Dict[str, Any]) -> str:
        """Records the outcome of a single game round.

        Args:
            final_scores (List[float]): Final cumulative scores of all players.
            outcomes (Dict[str, Any]): Dictionary to record wins, losses, and ties.

        Returns:
            str: Name of the winner or "tie" if there is no single winner.
        """
        # Check if all scores are equal (a tie)
        if all(score == final_scores[0] for score in final_scores):
            outcomes["ties"] += 1
            return "tie"

        # Find the maximum score and determine winners #TODO (lck: look into this -this is a bit confusing)
        max_score = max(final_scores)
        # Assume players in order "Player 1", "Player 2", etc.
        # This depends on the self.player_type keys (which must be in a stable order)
        # Identify winners/losers by mapping i -> player name
        # Suppose we match indexes to the order of self.llms.keys(), or define your own order #TODO: (lck: look into this)
        sorted_players = sorted(self.player_type.keys())  # or track your own ordering
        winners = [name for i, name in enumerate(sorted_players) if final_scores[i] == max_score]
        losers = [name for i, name in enumerate(sorted_players) if final_scores[i] != max_score]

        if len(winners) == 1:
            outcomes["wins"][winners[0]] += 1
            for loser in losers:
                outcomes["losses"][loser] += 1
            return winners[0]
        else:
            outcomes["ties"] += 1
            return "tie"

    def save_results(self, final_scores: List[float], state: Any) -> None:
        """Save simulation results to a JSON file."""
        results = self._prepare_results(state, final_scores)
        filename = self._get_results_filename()
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {filename}")

    def _prepare_results(self, state: Any, final_scores: List[float]) -> Dict[str, Any]:
        """Prepares the results dictionary for JSON serialization."""
        final_scores = list(final_scores)  # ensure it's a plain list
        return {
            "game_name": self.game_name,
            "final_state": str(state),
            "scores": self.scores,
            "returns": final_scores,
            "history": state.history_str(),
        }

    def _get_results_filename(self) -> str:
        """Generates the filename for saving results."""
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"{self.game_name.lower().replace(' ', '_')}_results.json")
