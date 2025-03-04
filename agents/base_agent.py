"""
base_agent.py

Defines a base class for all agents, which can then be subclassed by
HumanAgent, RandomAgent, LLMAgent, etc.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    filename="agent_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class BaseAgent(ABC):
    """Abstract base class for agents that pick actions in an OpenSpiel environment."""


    def __init__(self, agent_type: str = "generic"):
        """
        Initializes an agent with logging and performance tracking.

        Args:
            agent_type (str): The type of agent (e.g., "human", "llm", "random").
        """
        self.agent_type = agent_type  # The type of agent (e.g., "human", "llm", "random") Used for logging.
        self.action_count = 0
        self.total_time = 0

    @abstractmethod
    def compute_action(self, observation: Dict[str,Any]) -> int:
        """
        Selects an action based on the given observation.

        Args:
            observation (Dict[str, Any]): Contains game state, legal actions, etc.

        Returns:
            int: The action chosen by the agent.
        """
        pass

    def __call__(self, observation: Dict[str, Any]) -> int:
        """
        Allows the agent to be used as a callable function.

        Args:
            observation (Dict[str, Any]): Game state information.

        Returns:
            int: The action chosen.
        """
        return self._process_action(observation)

    def _process_action(self, observation: Dict[str, Any]) -> int:
            """Logs the observation, times the response, and calls `compute_action()`."""
            start_time = time.perf_counter()
            action = self.compute_action(observation)
            duration = time.perf_counter() - start_time

            self.action_count += 1
            self.total_time += duration

            logging.info(
                "[%s] Observation: %s, Action: %d, Time: %.6f seconds",
                self.agent_type, observation.get('state_string', 'N/A'), action, duration
            )

            return action

    def get_performance_metrics(self):
        """
        Returns performance statistics for the agent.

        Returns:
            Dict[str, float]: Action count, total time, and average response time.
        """
        avg_time = self.total_time / self.action_count if self.action_count > 0 else 0
        return {
            "agent_type": self.agent_type,
            "action_count": self.action_count,
            "total_processing_time": self.total_time,
            "average_response_time": avg_time
        }
