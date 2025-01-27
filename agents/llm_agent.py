"""
llm_agent.py

Implements an agent that uses an LLM to decide the next move.
"""

from typing import List, Any
from .base_agent import BaseAgent
from utils.llm_utils import generate_prompt, llm_decide_move

class LLMAgent(BaseAgent):
    """
    Agent that queries a language model (LLM) to pick an action.
    """

    def __init__(self, llm, game_name: str):
        """
        Args:
            llm (Any): The LLM instance (e.g., an OpenAI API wrapper, or any callable).
            game_name (str): The game's name for context in the prompt.
        """
        self.llm = llm
        self.game_name = game_name

    def compute_action(self, legal_actions: List[int], state: Any) -> int:
        """
        Uses the LLM to select an action from the legal actions.

        Args:
            legal_actions (List[int]): The set of legal actions for the current player.
            state (Any): The current OpenSpiel state.

        Returns:
            int: The action chosen by the LLM.
        """
        prompt = generate_prompt(self.game_name, str(state), legal_actions)
        return llm_decide_move(self.llm, prompt, tuple(legal_actions))

'''
 model_name=agent_cfg["model"],
                game=env.game,
                player_id=idx,
                temperature=agent_cfg.get("temperature", 0.7),
                max_tokens=agent_cfg.get("max_tokens", 128)

'''