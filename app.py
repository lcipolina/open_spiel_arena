import os
import json
import pandas as pd
import gradio as gr
from agents.llm_registry import LLM_REGISTRY  # Dynamically fetch LLM models
from simulators.tic_tac_toe_simulator import TicTacToeSimulator
from simulators.prisoners_dilemma_simulator import PrisonersDilemmaSimulator
from simulators.rock_paper_scissors_simulator import RockPaperScissorsSimulator
from games_registry import GAMES_REGISTRY
from simulators.base_simulator import PlayerType
from typing import Dict

# Extract available LLM models from the registry
llm_models = list(LLM_REGISTRY.keys())

# List of available games (manually defined for now)
games_list = [
    "rock_paper_scissors",
    "prisoners_dilemma",
    "tic_tac_toe",
    "connect_four",
    "matching_pennies",
    "kuhn_poker",
]

# Special leaderboard option for aggregating stats across all games
games_list.insert(0, "Total Performance")

# File to persist game results
RESULTS_TRACKER_FILE = "results_tracker.json"

# Load or initialize the results tracker
if os.path.exists(RESULTS_TRACKER_FILE):
    with open(RESULTS_TRACKER_FILE, "r") as f:
        results_tracker = json.load(f)
else:
    # Initialize tracking for all LLMs and games
    results_tracker = {
        llm: {game: {"games": 0, "moves/game": 0, "illegal-moves": 0,
                     "win-rate": 0, "vs Random": 0} for game in games_list[1:]}
        for llm in llm_models
    }

def save_results_tracker():
    """Save the results tracker to a JSON file."""
    with open(RESULTS_TRACKER_FILE, "w") as f:
        json.dump(results_tracker, f, indent=4)

def generate_stats_file(model_name: str) -> str:
    """Generate a JSON file with detailed statistics for the selected LLM model."""
    file_path = f"{model_name}_stats.json"
    with open(file_path, "w") as f:
        json.dump(results_tracker.get(model_name, {}), f, indent=4)
    return file_path

def provide_download_file(model_name):
    """Creates a downloadable JSON file with stats for the selected model."""
    return generate_stats_file(model_name)

def calculate_leaderboard(selected_game: str) -> pd.DataFrame:
    """
    Generate a structured leaderboard table.
    - If a specific game is selected, returns performance stats per LLM for that game.
    - If 'Total Performance' is selected, aggregates stats across all games.
    """
    leaderboard_df = pd.DataFrame(
        index=llm_models,
        columns=["# games", "moves/game", "illegal-moves", "win-rate", "vs Random"]
    )

    for llm in llm_models:
        if selected_game == "Total Performance":
            # Aggregate stats across all games
            total_games = sum(results_tracker[llm][game]["games"] for game in games_list[1:])
            total_moves = sum(results_tracker[llm][game]["moves/game"] * results_tracker[llm][game]["games"]
                              for game in games_list[1:])
            total_illegal_moves = sum(results_tracker[llm][game]["illegal-moves"] for game in games_list[1:])
            avg_win_rate = sum(results_tracker[llm][game]["win-rate"] * results_tracker[llm][game]["games"]
                               for game in games_list[1:]) / total_games if total_games > 0 else 0
            avg_vs_random = sum(results_tracker[llm][game]["vs Random"] * results_tracker[llm][game]["games"]
                                for game in games_list[1:]) / total_games if total_games > 0 else 0

            leaderboard_df.loc[llm] = [
                total_games,
                f"{(total_moves / total_games) if total_games > 0 else 0:.1f}",
                total_illegal_moves,
                f"{avg_win_rate:.1f}%",
                f"{avg_vs_random:.1f}%"
            ]
        else:
            # Retrieve stats for the selected game
            game_stats = results_tracker[llm].get(selected_game, {})
            leaderboard_df.loc[llm] = [
                game_stats.get("games", 0),
                game_stats.get("moves/game", 0),
                game_stats.get("illegal-moves", 0),
                f"{game_stats.get('win-rate', 0):.1f}%",
                f"{game_stats.get('vs Random', 0):.1f}%"
            ]

    leaderboard_df = leaderboard_df.reset_index()
    leaderboard_df.rename(columns={"index": "LLM Model"}, inplace=True)
    return leaderboard_df

def play_game(game_name, player1_type, player2_type, player1_model, player2_model, rounds):
    """Simulates a game session with the chosen players and logs results."""
    llms = {}
    if player1_type == "llm":
        llms["Player 1"] = player1_model
    if player2_type == "llm":
        llms["Player 2"] = player2_model

    simulator_class = GAMES_REGISTRY[game_name]
    simulator = simulator_class(game_name, llms=llms)
    game_states = []

    def log_fn(state):
        """Logs the current game state and available moves."""
        current_player = state.current_player()
        legal_moves = state.legal_actions(current_player)
        board = str(state)
        game_states.append(f"Current Player: {current_player}\nBoard:\n{board}\nLegal Moves: {legal_moves}")

    results = simulator.simulate(rounds=int(rounds), log_fn=log_fn)
    return "\n".join(game_states) + f"\nGame Result: {results}"

# Gradio Interface
with gr.Blocks() as interface:
    # Game Arena Tab
    with gr.Tab("Game Arena"):
        gr.Markdown("# LLM Game Arena\nSelect a game and players to play against LLMs.")

        game_dropdown = gr.Dropdown(choices=games_list[1:], label="Select a Game", value=games_list[1])
        player1_dropdown = gr.Dropdown(choices=["human", "random_bot", "llm"], label="Player 1 Type", value="llm")
        player2_dropdown = gr.Dropdown(choices=["human", "random_bot", "llm"], label="Player 2 Type", value="random_bot")
        player1_model_dropdown = gr.Dropdown(choices=llm_models, label="Player 1 Model", visible=False)
        player2_model_dropdown = gr.Dropdown(choices=llm_models, label="Player 2 Model", visible=False)
        rounds_slider = gr.Slider(1, 10, step=1, label="Rounds")
        result_output = gr.Textbox(label="Game Result")

        play_button = gr.Button("Play Game")
        play_button.click(
            play_game,
            inputs=[game_dropdown, player1_dropdown, player2_dropdown, player1_model_dropdown, player2_model_dropdown, rounds_slider],
            outputs=result_output,
        )

    # Leaderboard Tab
    with gr.Tab("Leaderboard"):
        gr.Markdown("# LLM Model Leaderboard\nTrack performance across different games!")

        game_dropdown = gr.Dropdown(choices=games_list, label="Select Game", value="Total Performance")
        leaderboard_table = gr.Dataframe(value=calculate_leaderboard("Total Performance"), label="Leaderboard")
        model_dropdown = gr.Dropdown(choices=llm_models, label="Select LLM Model")
        download_button = gr.File(label="Download Statistics File")
        refresh_button = gr.Button("Refresh Leaderboard")

        def update_leaderboard(selected_game):
            """Updates the leaderboard based on the selected game."""
            return calculate_leaderboard(selected_game)

        model_dropdown.change(fn=provide_download_file, inputs=[model_dropdown], outputs=[download_button])
        game_dropdown.change(fn=update_leaderboard, inputs=[game_dropdown], outputs=[leaderboard_table])
        refresh_button.click(fn=update_leaderboard, inputs=[game_dropdown], outputs=[leaderboard_table])

interface.launch()
