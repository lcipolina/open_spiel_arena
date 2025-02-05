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
        llm: {game: {"wins": 0, "ties": 0, "losses": 0, "games": 0} for game in games_list}
        for llm in llm_models
    }


def save_results_tracker():
    """Save the results tracker to a JSON file."""
    with open(RESULTS_TRACKER_FILE, "w") as f:
        json.dump(results_tracker, f, indent=4)


def calculate_leaderboard():
    """Generate a structured leaderboard table summarizing LLM performance across games."""

    # Create a DataFrame where rows are LLMs and columns are games
    leaderboard_df = pd.DataFrame(index=llm_models, columns=games_list)

    for llm in llm_models:
        for game in games_list:
            games_played = max(1, results_tracker[llm][game]['games'])  # Avoid division by zero
            wins = (results_tracker[llm][game]['wins'] / games_played) * 100
            ties = (results_tracker[llm][game]['ties'] / games_played) * 100
            losses = (results_tracker[llm][game]['losses'] / games_played) * 100

            # Format as percentage string
            leaderboard_df.loc[llm, game] = f"{wins:.1f}% W / {ties:.1f}% T / {losses:.1f}% L"

    # Ensure LLM names appear in the first column
    leaderboard_df = leaderboard_df.reset_index()
    leaderboard_df.rename(columns={"index": "LLM Model"}, inplace=True)

    return leaderboard_df


def get_model_details(model_name):
    """Returns detailed performance breakdown of the selected LLM model."""
    if model_name not in results_tracker:
        return "No data available for this model."

    details = f"### {model_name} Performance Breakdown\n"
    for game, record in results_tracker[model_name].items():
        total_games = record["games"]
        details += (
            f"- **{game.capitalize()}**: {record['wins']} Wins, "
            f"{record['ties']} Ties, {record['losses']} Losses (Total: {total_games})\n"
        )

    return details


# Gradio Interface
with gr.Blocks() as interface:
    with gr.Tab("Game Arena"):
        gr.Markdown("# LLM Game Arena\nPlay against LLMs or other players in classic games!")

        # (Game selection and play functionality remains unchanged)

    with gr.Tab("Leaderboard"):
        gr.Markdown("# LLM Model Leaderboard\nTrack performance across different games!")

        leaderboard_table = gr.Dataframe(value=calculate_leaderboard(), label="Leaderboard")

        with gr.Row():
            model_dropdown = gr.Dropdown(choices=llm_models, label="Select LLM Model")
        details_output = gr.Textbox(label="Model Performance Details", interactive=False)

        def update_leaderboard():
            """Updates the leaderboard table."""
            return calculate_leaderboard()

        def update_details(model_name):
            """Updates the details section when an LLM is selected."""
            return get_model_details(model_name)

        update_leaderboard_button = gr.Button("Refresh Leaderboard")
        update_leaderboard_button.click(fn=update_leaderboard, inputs=[], outputs=[leaderboard_table])

        model_dropdown.change(fn=update_details, inputs=[model_dropdown], outputs=[details_output])

interface.launch()
