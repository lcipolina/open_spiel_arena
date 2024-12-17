# scripts/run_simulation.py
"""Main entry point for running game simulations.

This script serves as the main binary for executing simulations of various games
using OpenSpiel and Large Language Models (LLMs). It supports command-line
arguments to specify the game and simulation configurations.
"""

import os
import argparse
from transformers import pipeline
from games_registry import GAMES_REGISTRY # Import the game loaders available


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
        game_info = GAMES_REGISTRY[game_name]

        # Load the game and create the simulator dynamically
        game_instance = game_info["loader"]()
        simulator_class = game_info["simulator"]
        simulator = simulator_class(game_instance, game_info["display_name"], llms)


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
