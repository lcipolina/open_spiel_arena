# simulators/base_simulator.py
"""Base class for game simulators.

This module defines the GameSimulator abstract base class, which serves as the
foundation for implementing specific game simulators. Each simulator inherits
from this class and implements the `simulate` method for game-specific logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class GameSimulator(ABC):
    """Abstract base class for game simulators.

    Args:
        game: The game object from OpenSpiel.
        game_name: The name of the game being simulated.
        llms: A dictionary mapping model names to LLM instances.
        random_bot: Whether to include a random bot.
        play_against_itself: Whether the LLM plays against itself.
    """

    def __init__(self, game: Any, game_name: str, llms: Dict[str, Any],
                 random_bot: bool = False, play_against_itself: bool = False) -> None:
        self.game = game
        self.game_name = game_name
        self.llms = llms
        self.random_bot = random_bot
        self.play_against_itself = play_against_itself

    @abstractmethod
    def simulate(self) -> Dict[str, int]:
        """Simulates the game.

        Returns:
            Dict[str, int]: The scores for each player.
        """
        pass


