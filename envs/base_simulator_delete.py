'''Base class for game-specific simulators.'''

#TODO: delete this!
'''
from typing import Any, Dict, List
from abc import ABC, abstractmethod

class GameSimulator(ABC):
    """Base class for game-specific simulators."""

    @abstractmethod
    def get_observation(self, state) -> Dict[str, Any]:
        """Generate the observation from the game state."""
        pass

    @abstractmethod
    def get_action(self, state, actions: List[int]):
        """Apply actions to the state."""
        pass
'''
