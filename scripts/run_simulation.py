import argparse
from games_registry import GAMES_REGISTRY
from simulators.base_simulator import PlayerType

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Run OpenSpiel simulations.")
    parser.add_argument(
        "--games",
        type=str,
        nargs="+",
        required=True,
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
        "--player-type",
        type=str,
        choices=["human", "random_bot", "llm", "self_play"],
        default="llm",
        help="Type of player for the simulation."
    )
    args = parser.parse_args()

    # Convert player type to enum
    player_type = PlayerType(args.player_type)

    # Initialize overall leaderboard
    overall_leaderboard = {}

    # Loop through selected games
    for game_name in args.games:
        game_config = GAMES_REGISTRY[game_name]
        game = game_config["loader"]()
        simulator_class = game_config["simulator"]

        simulator = simulator_class(
            game,
            game_config["display_name"],
            llms={},  # Pass LLM configurations here if needed
            player_type=player_type
        )

        print(f"\nStarting simulation for {game_name} with player type: {args.player_type}...")
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
