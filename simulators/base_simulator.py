import os
import json
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import random
from utils.llm_utils import generate_prompt, llm_decide_move
from enum import Enum


class PlayerType(Enum):
    HUMAN = "human"
    RANDOM_BOT = "random_bot"
    LLM = "llm"
    SELF_PLAY = "self_play"


class GameSimulator(ABC):
    """Base class for simulating games with LLMs.

    Handles common functionality like state transitions, scoring, and logging.
    """

    def __init__(self, game: Any, game_name: str, llms: Dict[str, Any],
                 player_type: PlayerType = PlayerType.LLM):
        self.game = game
        self.game_name = game_name
        self.llms = llms
        self.player_type = player_type
        self.scores = {name: 0 for name in self.llms.keys()}  # Initialize scores

    def simulate(self, rounds: int = 1, log_fn=None) -> Dict[str, Any]:
        """Simulates the game for multiple rounds.

        Args:
            rounds: Number of times the game should be played.
            log_fn: Optional function to log intermediate states.

        Returns:
            Dict[str, Any]: Summary of results for all rounds.
        """
        outcomes = self._initialize_outcomes() # Reset the outcomes dictionary

        for _ in range(rounds):
            self.scores = {name: 0 for name in self.llms.keys()}  # Reset scores
            state = self.game.new_initial_state()

            while not state.is_terminal():
                if log_fn:
                    log_fn(state)  # Log current state if a logging function is provided

                current_player = state.current_player()
                if current_player < 0:
                    self._apply_default_action(state)
                    continue

                legal_actions = state.legal_actions(current_player)
                action = self._get_action(current_player, state, legal_actions)
                state.apply_action(action)

            # Record outcomes
            final_scores = state.returns()
            self._record_outcomes(final_scores, outcomes)

        return outcomes

    def _collect_actions(self, state: Any) -> List[int]:
        """Collects actions for all players in a simultaneous-move game.

        Args:
            state: The current game state.

        Returns:
            List[int]: Actions chosen by all players.
        """
        return [
            self._get_action(player, state, state.legal_actions(player))
            for player in range(self.game.num_players())
        ]

    def _initialize_outcomes(self) -> Dict[str, Any]:
        """Initializes the outcomes dictionary."""
        return {"wins": {name: 0 for name in self.llms.keys()},
                "losses": {name: 0 for name in self.llms.keys()},
                "ties": 0}

    def _get_action(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Gets the action for the current player.

        Args:
            player: The index of the current player.
            state: The current game state.
            legal_actions: The legal actions available for the player.

        Returns:
            int: The action selected by the player.
        """
        if self.player_type == PlayerType.HUMAN and player == 0:
            return self._get_human_action(state, legal_actions)

        if self.player_type == PlayerType.RANDOM_BOT and player == 1:
            return random.choice(legal_actions)

        if self.player_type in {PlayerType.SELF_PLAY, PlayerType.LLM} and player < len(self.llms):
            return self._get_llm_action(player, state, legal_actions)

        return legal_actions[0]  # Default fallback action

    def _get_human_action(self, state: Any, legal_actions: List[int]) -> int:
        """Handles input for human players."""
        print(f"Current state of {self.game_name}:\n{state}")
        print(f"Your options: {legal_actions}") # Display legal moves to the user
        while True:
            try:
                action = int(input("Enter your action (number): "))
                if action in legal_actions: # Validate the move
                    return action
            except ValueError:
                pass
            print("Invalid action. Please choose from:", legal_actions)

    def _get_llm_action(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Handles LLM-based decisions."""
        model_name = list(self.llms.keys())[player % len(self.llms)]
        llm = self.llms[model_name]
        prompt = generate_prompt(self.game_name, str(state), legal_actions)
        return llm_decide_move(llm, prompt, tuple(legal_actions))


    def _apply_default_action(self, state):
        """
        Applies a default action when the current player is invalid.
        """
        state.apply_action(state.legal_actions()[0])

    def _record_outcomes(self, final_scores: List[float], outcomes: Dict[str, Any]) -> str:
        """Records the outcome of a single game round."""
        if all(score == final_scores[0] for score in final_scores):
            outcomes["ties"] += 1
            return "tie"

        max_score = max(final_scores)
        winners = [name for i, name in enumerate(self.llms.keys()) if final_scores[i] == max_score]

        if len(winners) == 1:
            outcomes["wins"][winners[0]] += 1
            return winners[0]
        else:
            outcomes["ties"] += 1
            return "tie"

    def save_results(self, state: Any, final_scores: List[float]) -> None:
        """Save simulation results to a JSON file."""
        results = self._prepare_results(state, final_scores)
        filename = self._get_results_filename()

        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {filename}")

    def _prepare_results(self, state: Any, final_scores: List[float]) -> Dict[str, Any]:
        """Prepares the results dictionary for JSON serialization."""
        final_scores = final_scores.tolist() if hasattr(final_scores, "tolist") else final_scores
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

    def log_progress(self, state: Any) -> None:
        """Log the current game state."""
        print(f"Current state of {self.game_name}:\n{state}")
