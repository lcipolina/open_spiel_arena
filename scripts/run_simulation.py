# scripts/run_simulation.py
"""Main entry point for running game simulations.

This script serves as the main binary for executing simulations of various games
using OpenSpiel and Large Language Models (LLMs). It supports command-line
arguments to specify the game and simulation configurations.
"""

import os
import argparse
from transformers import pipeline
from simulators.tic_tac_toe_simulator import TicTacToeSimulator
from simulators.prisoners_dilemma_simulator import PrisonersDilemmaSimulator
from simulators.rock_paper_scissors_simulator import RockPaperScissorsSimulator
from games.tic_tac_toe import get_tic_tac_toe_game
from games.prisoners_dilemma import get_prisoners_dilemma_game
from games.rock_paper_scissors import get_rps_game

# Force Hugging Face Transformers to use PyTorch backend instead of TensorFlow
os.environ["TRANSFORMERS_BACKEND"] = "pt"


def main() -> None:
    """Main function for running game simulations."""
    parser = argparse.ArgumentParser(description="Run OpenSpiel simulations.")
    parser.add_argument(
        "--games",
        type=str,
        nargs="+",
        choices=["tic_tac_toe", "prisoners_dilemma", "rps"],
        required=True,
        help="The list of games to simulate (e.g., tic_tac_toe rps)."
    )
    args = parser.parse_args()

    # Load LLMs
    llms = {
    "google/flan-t5-small": pipeline("text2text-generation", model="google/flan-t5-small"),
    "gpt2": pipeline("text-generation", model="gpt2"),
}

    # Initialize leaderboard
    overall_leaderboard = {name: 0.0 for name in llms.keys()}

    # Simulate each selected game
    for game_name in args.games:
        print(f"\nStarting simulation for {game_name}...")

        # Select the game and simulator
        if game_name == "tic_tac_toe":
            game = get_tic_tac_toe_game()
            simulator = TicTacToeSimulator(game, "Tic-Tac-Toe", llms, random_bot=True)
        elif game_name == "prisoners_dilemma":
            game = get_prisoners_dilemma_game()
            simulator = PrisonersDilemmaSimulator(
                game, "Iterated Prisoner's Dilemma", llms, play_against_itself=True, max_iterations=10
            )
        elif game_name == "rps":
            game = get_rps_game()
            simulator = RockPaperScissorsSimulator(game, "Rock-Paper-Scissors", llms)
        else:
            print(f"Unsupported game: {game_name}")
            continue

        # Run the simulation and update leaderboard
        game_results = simulator.simulate()
        for llm, score in game_results.items():
            overall_leaderboard[llm] += score

    # Display final leaderboard
    print("\nOverall Leaderboard:")
    for model_name, total_score in overall_leaderboard.items():
        print(f"{model_name}: {total_score}")


if __name__ == "__main__":
    main()
