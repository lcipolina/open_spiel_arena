# tests/test_simulators.py
"""Unit tests for game simulators.

This script tests the functionality of various game simulators, ensuring that
they behave correctly under normal and edge-case scenarios.
"""

import unittest
from unittest.mock import MagicMock
from open_spiel_simulation.simulators.tic_tac_toe_simulator import TicTacToeSimulator
from open_spiel_simulation.simulators.prisoners_dilemma_simulator import PrisonersDilemmaSimulator
from open_spiel_simulation.simulators.rock_paper_scissors_simulator import RockPaperScissorsSimulator
from open_spiel_simulation.games.tic_tac_toe import get_tic_tac_toe_game
from open_spiel_simulation.games.prisoners_dilemma import get_prisoners_dilemma_game
from open_spiel_simulation.games.rock_paper_scissors import get_rps_game

class TestSimulators(unittest.TestCase):
    """Unit tests for game simulators."""

    def setUp(self) -> None:
        """Set up test cases with mock LLMs and games."""
        self.mock_llm = MagicMock()
        self.llms = {"mock_model": self.mock_llm}

    def test_tic_tac_toe_simulator(self):
        """Test Tic-Tac-Toe simulator functionality."""
        game = get_tic_tac_toe_game()
        simulator = TicTacToeSimulator(game, "Tic-Tac-Toe", self.llms, random_bot=True)

        self.mock_llm.return_value = [{"generated_text": "0"}]  # Mock LLM decision
        results = simulator.simulate()

        self.assertIsInstance(results, dict)
        self.assertIn("mock_model", results)

    def test_prisoners_dilemma_simulator(self):
        """Test Prisoner's Dilemma simulator functionality."""
        game = get_prisoners_dilemma_game()
        simulator = PrisonersDilemmaSimulator(
            game, "Iterated Prisoner's Dilemma", self.llms, play_against_itself=True, max_iterations=5
        )

        self.mock_llm.return_value = [{"generated_text": "0"}]  # Mock LLM decision
        results = simulator.simulate()

        self.assertIsInstance(results, dict)
        self.assertIn("mock_model", results)

    def test_rock_paper_scissors_simulator(self):
        """Test Rock-Paper-Scissors simulator functionality."""
        game = get_rps_game()
        simulator = RockPaperScissorsSimulator(game, "Rock-Paper-Scissors", self.llms)

        self.mock_llm.return_value = [{"generated_text": "0"}]  # Mock LLM decision
        results = simulator.simulate()

        self.assertIsInstance(results, dict)
        self.assertIn("mock_model", results)

if __name__ == "__main__":
    unittest.main()

