import os
import json
import pandas as pd
import gradio as gr
from agents.llm_registry import LLM_REGISTRY  # Dynamically fetch LLM models

# Extract available LLM models
llm_models = list(LLM_REGISTRY.keys())

# Define game list manually (for now)
games_list = [
    "rock_paper_scissors",
    "prisoners_dilemma",
    "tic_tac_toe",
    "connect_four",
    "matching_pennies",
    "kuhn_poker",
]

# File to persist results
RESULTS_TRACKER_FILE = "results_tracker.json"

# Load or initialize the results tracker
if os.path.exists(RESULTS_TRACKER_FILE):
    with open(RESULTS_TRACKER_FILE, "r") as f:
        results_tracker = json.load(f)
else:
    results_tracker = {
        llm: {game: {"games": 0, "moves/game": 0, "illegal-moves": 0,
                     "win-rate": 0, "vs Random": 0} for game in games_list}
        for llm in llm_models
    }

def save_results_tracker():
    """Save the results tracker to a JSON file."""
    with open(RESULTS_TRACKER_FILE, "w") as f:
        json.dump(results_tracker, f, indent=4)

def calculate_leaderboard(selected_game: str) -> pd.DataFrame:
    """Generate a structured leaderboard table for the selected game."""
    leaderboard_df = pd.DataFrame(index=llm_models,
                                  columns=["# games", "moves/game",
                                           "illegal-moves", "win-rate", "vs Random"])

    for llm in llm_models:
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

def generate_stats_file(model_name: str):
    """Generate a JSON file with detailed statistics for the selected LLM model."""
    file_path = f"{model_name}_stats.json"
    with open(file_path, "w") as f:
        json.dump(results_tracker.get(model_name, {}), f, indent=4)
    return file_path

# Gradio Interface
with gr.Blocks() as interface:
    with gr.Tab("Game Arena"):
        gr.Markdown("# LLM Game Arena\nPlay against LLMs or other players in classic games!")

    with gr.Tab("Leaderboard"):
        gr.Markdown("# LLM Model Leaderboard\nTrack performance across different games!")

        game_dropdown = gr.Dropdown(choices=games_list, label="Select Game", value=games_list[0])
        leaderboard_table = gr.Dataframe(value=calculate_leaderboard(games_list[0]), label="Leaderboard")
        model_dropdown = gr.Dropdown(choices=llm_models, label="Select LLM Model")
        download_button = gr.File(label="Download Statistics File")
        refresh_button = gr.Button("Refresh Leaderboard")

        def update_leaderboard(selected_game):
            """Updates the leaderboard table based on the selected game."""
            return calculate_leaderboard(selected_game)

        def provide_download_file(model_name):
            """Creates a downloadable JSON file with stats for the selected model."""
            return generate_stats_file(model_name)

        def refresh_leaderboard():
            """Manually refresh the leaderboard."""
            return calculate_leaderboard(game_dropdown.value)

        game_dropdown.change(fn=update_leaderboard, inputs=[game_dropdown], outputs=[leaderboard_table])
        model_dropdown.change(fn=provide_download_file, inputs=[model_dropdown], outputs=[download_button])
        refresh_button.click(fn=refresh_leaderboard, inputs=[], outputs=[leaderboard_table])

interface.launch()
