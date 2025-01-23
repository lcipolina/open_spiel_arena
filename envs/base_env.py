"""
base_env.py

Defines a Gym-like abstract environment interface.
"""

from typing import Dict, Any, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum, unique

@unique
class PlayerId(Enum):
    CHANCE = -1
    SIMULTANEOUS = -2
    INVALID = -3
    TERMINAL = -4
    MEAN_FIELD = -5

    @classmethod
    def from_value(cls, value: int):
        """Returns the PlayerId corresponding to a given integer value.

        Args:
            value (int): The numerical value to map to a PlayerId.

        Returns:
            PlayerId: The matching enum member, or raises a ValueError if invalid.
        """
        for member in cls:
            if member.value == value:
                return member
        if value >= 0:  # Positive integers represent default players
            return None  # No enum corresponds to these values directly
        raise ValueError(f"Unknown player ID value: {value}")


class PlayerType(Enum):
    HUMAN = "human"
    RANDOM_BOT = "random_bot"
    LLM = "llm"
    SELF_PLAY = "self_play"

class BaseEnv(ABC):
    """
    A Gym-like abstract base class.

    We also include a few utility methods that are relevant in the
    OpenSpiel context (chance node handling, outcomes recording, etc.).
    """

    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment to an initial state and return an initial observation."""
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Applies an action to the environment.

        Returns:
            observation (Any): The next observation after applying the action.
            reward (float): The reward for this step.
            done (bool): Whether the episode is finished.
            info (dict): Additional diagnostic information.
        """
        pass

    def render(self, mode: str = 'human') -> None:
        """Optional: Print or visualize the current state."""
        pass

    def close(self) -> None:
        """Optional: Perform any necessary cleanup."""
        pass

    def normalize_player_id(self, player_id: Union[int, PlayerId]) -> int:
        """Normalize player_id to its integer value for consistent comparisons.

           This is needed as OpenSpiel has ambiguous representation of the playerID

        Args:
            player_id (Union[int, PlayerId]): The player ID, which can be an
                integer or a PlayerId enum instance.

        Returns:
            int: The integer value of the player ID.
        """
        if isinstance(player_id, PlayerId):
            return player_id.value  # Extract the integer value from the enum
        return player_id  # If already an integer, return it as is
