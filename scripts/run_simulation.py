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
from simulators.base_simulator import PlayerType


# Force Hugging Face Transformers to use PyTorch backend instead of TensorFlow
os.environ["TRANSFORMERS_BACKEND"] = "pt"


def main() -> None:
    """Main function for running game simulations."""
    parser = argparse.ArgumentParser(description="Run OpenSpiel simulations.")
   # Dynamically extract available games from the registry
    available_games = list(GAMES_REGISTRY.keys())

    parser.add_argument(
        "--games",
        type=str,
        choices=available_games,
        required=True,
        nargs="+",
        help=f"The name(s) of the game(s) to simulate. Available options: {', '.join(available_games)}",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds to play for each game."
    )
    parser.add_argument(
        "--player-type",
        type=str,
        choices=["human", "random_bot", "llm", "self_play"],
        default="llm",
        help="Type of player for the simulation (default: llm)."
    )
    args = parser.parse_args()

    # Convert player type to enum
    player_type = PlayerType(args.player_type)

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
        simulator = simulator_class(game_instance,
                                    game_info["display_name"],
                                    llms,
                                    player_type=player_type
                                    )

        print(f"Starting simulation for {game_name} with player type: {args.player_type}...")

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
