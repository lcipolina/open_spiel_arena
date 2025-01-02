''' Script to run simulations for OpenSpiel games. '''

import argparse
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games_registry import GAMES_REGISTRY
from llm_registry import LLM_REGISTRY
from simulators.base_simulator import PlayerType


def _resolve_llms(player1_type, player2_type, player1_model, player2_model):
    """Resolve the LLM configuration for both players.

    Args:
        player1_type (str): Type of Player 1 ("human", "random_bot", "llm").
        player2_type (str): Type of Player 2 ("human", "random_bot", "llm").
        player1_model (str): Selected LLM for Player 1 (if applicable).
        player2_model (str): Selected LLM for Player 2 (if applicable).

    Returns:
        dict: A dictionary mapping player indices to LLMs.
    """
    llms = {}
    available_llms = list(LLM_REGISTRY.keys())

    if player1_type == "llm":
        if player1_model:
            llms["Player 1"] = LLM_REGISTRY[player1_model]["model_loader"]()
        else:
            print("Player 1 LLM not specified. Randomly assigning an LLM.")
            llms["Player 1"] = LLM_REGISTRY[random.choice(available_llms)]["model_loader"]()

    if player2_type == "llm":
        if player2_model:
            llms["Player 2"] = LLM_REGISTRY[player2_model]["model_loader"]()
        else:
            print("Player 2 LLM not specified. Randomly assigning an LLM.")
            llms["Player 2"] = LLM_REGISTRY[random.choice(available_llms)]["model_loader"]()

    return llms


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Run OpenSpiel simulations.")
    parser.add_argument(
        "--games",
        type=str,
        nargs="+",
        default=["tic_tac_toe"],  # Default game
        choices=list(GAMES_REGISTRY.keys()),
        help="The games to simulate."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds to play for each game."
    )
    parser.add_argument(
        "--player1-type",
        type=str,
        choices=["human", "random_bot", "llm"],
        default="llm",
        help="Type of Player 1 (human, random_bot, or llm)."
    )
    parser.add_argument(
        "--player2-type",
        type=str,
        choices=["human", "random_bot", "llm"],
        default="llm",
        help="Type of Player 2 (human, random_bot, or llm)."
    )
    parser.add_argument(
        "--player1-model",
        type=str,
        default=list(LLM_REGISTRY.keys())[0],
        choices=list(LLM_REGISTRY.keys()),
        help="Specific LLM model for Player 1 (optional)."
    )
    parser.add_argument(
        "--player2-model",
        type=str,
        default=list(LLM_REGISTRY.keys())[0],
        choices=list(LLM_REGISTRY.keys()),
        help="Specific LLM model for Player 2 (optional)."
    )
    args = parser.parse_args()

    # Convert player types to enums
    player1_type = args.player1_type
    player2_type = args.player2_type

    # Initialize overall leaderboard
    overall_leaderboard = {}

    # Loop through selected games
    for game_name in args.games:
        game_config = GAMES_REGISTRY[game_name]
        game = game_config["loader"]()
        simulator_class = game_config["simulator"]

        # Resolve LLMs
        llms = _resolve_llms(player1_type, player2_type, args.player1_model, args.player2_model)

        # Initialize simulator
        simulator = simulator_class(
            game,
            game_config["display_name"],
            llms=llms,
            player_type={"Player 1": player1_type, "Player 2": player2_type}
        )

        print(f"\nStarting simulation for {game_name}...")
        game_results = simulator.simulate(rounds=args.rounds)

        # Update leaderboard
        for player, score in game_results["wins"].items():
            overall_leaderboard[player] = overall_leaderboard.get(player, 0) + score

    # Print overall leaderboard
    print("\nOverall Leaderboard:")
    for player, score in overall_leaderboard.items():
        print(f"{player}: {score}")


if __name__ == "__main__":
    main()
