import os
import json
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import random
from utils.llm_utils import generate_prompt, llm_decide_move

class GameSimulator(ABC):
    """Base class for simulating games with LLMs.

    Handles common functionality like state transitions, scoring, and logging.
    """

    def __init__(self, game: Any, game_name: str, llms: Dict[str, Any],
                 random_bot: bool = False, play_against_itself: bool = False):
        self.game = game
        self.game_name = game_name
        self.llms = llms
        self.random_bot = random_bot
        self.play_against_itself = play_against_itself
        self.scores = {name: 0 for name in self.llms.keys()}  # Initialize scores

    @abstractmethod
    def simulate(self) -> Dict[str, int]:
        """Simulates the game. To be overridden by subclasses."""
        pass

    @staticmethod
    def update_leaderboard(leaderboard: Dict[str, float], scores: Dict[str, float]) -> Dict[str, float]:
        """Update the leaderboard with scores from a single game.

        Args:
            leaderboard: A dictionary of cumulative scores for LLMs.
            scores: A dictionary of scores from the current game.

        Returns:
            Dict[str, float]: Updated leaderboard.
        """
        for llm, score in scores.items():
            leaderboard[llm] = leaderboard.get(llm, 0) + score
        return leaderboard


    def save_results(self, state: Any, final_scores: List[float]) -> None:
        """Save simulation results to a JSON file.

        Args:
            state: The final game state.
            final_scores: The final scores for each player.
        """
        # Ensure results directory exists
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(
            results_dir, f"{self.game_name.lower().replace(' ', '_')}_results.json"
        )

        # Convert NumPy arrays to lists for JSON serialization
        final_scores = final_scores.tolist() if hasattr(final_scores, "tolist") else final_scores

        results = {
            "game_name": self.game_name,
            "final_state": str(state),
            "scores": self.scores,
            "returns": final_scores,
            "history": state.history_str(),
        }

        # Save results to a JSON file
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {filename}")


    def log_progress(self, state: Any) -> None:
        """Log the current game state."""
        print(f"Current state of {self.game_name}:\n{state}")


    def _get_action(self, player: int, state: Any, legal_actions: List[int]) -> int:
        """Gets the action for the current player.

        Args:
            player: The index of the current player.
            state: The current game state.
            legal_actions: The legal actions available for the player.

        Returns:
            int: The action selected by the player.
        """
        if self.random_bot and player == 1:
            return random.choice(legal_actions)
        elif self.play_against_itself:
            model_name = list(self.llms.keys())[player % len(self.llms)]
            llm = self.llms[model_name]
            prompt = generate_prompt(self.game_name, str(state), legal_actions)
            return llm_decide_move(llm, prompt, tuple(legal_actions))  # Convert to tuple
        elif player < len(self.llms):
            model_name = list(self.llms.keys())[player]
            llm = self.llms[model_name]
            prompt = generate_prompt(self.game_name, str(state), legal_actions)
            return llm_decide_move(llm, prompt, tuple(legal_actions))  # Convert to tuple
        return legal_actions[0]  # Default fallback action
