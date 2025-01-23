''' Script to run simulations for OpenSpiel games. '''

import argparse
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games_registry import GAMES_REGISTRY
from llm_registry import LLM_REGISTRY
from simulators.main_simulator import PlayerType


def _resolve_llms(player_types, player_models):
    """Assign the LLM configuration for all players.

    Args:
        player_types (List[str]): Types of players ("human", "random_bot", "llm").
        player_models (List[str]): Selected LLMs for players (if applicable).

    Returns:
        dict: A dictionary mapping player names to LLMs.
    """
    llms = {}
    available_llms = list(LLM_REGISTRY.keys())

    for i, player_type in enumerate(player_types):
        player_name = f"Player {i + 1}"
        if player_type == "llm":
            if player_models[i]:
                llms[player_name] = LLM_REGISTRY[player_models[i]]["model_loader"]()
            else:
                print(f"{player_name} LLM not specified. Randomly assigning an LLM.")
                llms[player_name] = LLM_REGISTRY[random.choice(available_llms)]["model_loader"]()

    return llms


def main():
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
        "--player-types",
        type=str,
        nargs="+",
        choices=["human", "random_bot", "llm"],
        default=["llm", "llm"],
        help="Types of players (e.g., human, random_bot, llm)."
    )
    parser.add_argument(
        "--player-models",
        type=str,
        nargs="+",
        default=[],
        help="Specific LLM models for players (optional)."
    )
    args = parser.parse_args()

    overall_leaderboard = {}

    for game_name in args.games:
        game_config = GAMES_REGISTRY[game_name]
        game = game_config["loader"]()
        simulator_class = game_config["simulator"]

        # Get the required number of players for this game
        num_players = game.num_players()

        # Adjust player types dynamically
        player_types = args.player_types[:num_players]  # Trim excess if too many
        while len(player_types) < num_players:
            player_types.append("random_bot")  # Default to random_bot for missing players

        # Adjust player models dynamically
        player_models = args.player_models[:num_players]  # Trim excess if too many
        while len(player_models) < num_players:
            player_models.append(None)  # Default to None for missing models

        # Resolve LLMs
        llms = _resolve_llms(player_types, player_models)

        # Map player types to player names
        player_type_map = {f"Player {i + 1}": player_types[i] for i in range(num_players)}

        # Initialize simulator
        simulator = simulator_class(
            game,
            game_config["display_name"],
            llms=llms,
            player_type=player_type_map,
        )

        print(f"\nStarting simulation for {game_name}...")
        game_results = simulator.simulate(rounds=args.rounds)

        for player, score in game_results["wins"].items():
            overall_leaderboard[player] = overall_leaderboard.get(player, 0) + score

    # Print overall leaderboard
    print("\nOverall Leaderboard:")
    for player, score in overall_leaderboard.items():
        print(f"{player}: {score}")




if __name__ == "__main__":
    main()
