"""
app.py

Script to interact with the game simulator using Gradio.
"""

import os
import json
import gradio as gr
from games_registry import GAMES_REGISTRY  #THIS DOESN'T EXIST ANYMORE, for now, we can have a manual list of the games and then we see how we can make it dynamic
from agents.llm_registry import LLM_REGISTRY

games_list = ["rock_paper_scissors","prisoners_dilemma", "tic_tac_toe", "connect_four","matching_pennies", "kuhn_poker"]

# File to persist results
RESULTS_TRACKER_FILE = "results_tracker.json"

# Load or initialize the results tracker
if os.path.exists(RESULTS_TRACKER_FILE):
    with open(RESULTS_TRACKER_FILE, "r") as f:
        results_tracker = json.load(f)
else:
    results_tracker = {
    name: {opponent: {"wins": 0, "games": 0} for opponent in ["Human"] + list(LLM_REGISTRY.keys())}
                       for name in ["Human"] + list(LLM_REGISTRY.keys())
                       }


def save_results_tracker():
    """Save the results tracker to a JSON file."""
    with open(RESULTS_TRACKER_FILE, "w") as f:
        json.dump(results_tracker, f, indent=4)


def initialize_game(game_name, player1_type, player2_type, player1_model, player2_model):
    """Initialize the game state and simulator."""
    game_config = GAMES_REGISTRY[game_name]
    game = game_config["loader"]()
    simulator_class = game_config["simulator"]

    # Ensure models are selected if players are LLMs
    if player1_type == "llm" and not player1_model:
        raise ValueError("Player 1 is set to LLM, but no model is selected.")
    if player2_type == "llm" and not player2_model:
        raise ValueError("Player 2 is set to LLM, but no model is selected.")

    # Initialize LLMs for the players
    llms = {
        "Player 1": LLM_REGISTRY[player1_model]["model_loader"]() if player1_type == "llm" else None,
        "Player 2": LLM_REGISTRY[player2_model]["model_loader"]() if player2_type == "llm" else None,
    }

    # Map player types to names
    player_type_map = {
        "Player 1": player1_type,
        "Player 2": player2_type,
    }

    # Create the simulator
    simulator = simulator_class(
        game,
        game_name,
        llms=llms,
        player_type=player_type_map,
    )
    state = game.new_initial_state()

    return simulator, state, "Game Initialized! Click 'Next Turn' to start."


def toggle_model_dropdown(player1, player2):
    """Control visibility and set default models for LLM players."""
    player1_model_visible = gr.update(visible=(player1 == "llm"))
    player2_model_visible = gr.update(visible=(player2 == "llm"))

    # Set default models if the player type is "llm"
    default_model1 = list(LLM_REGISTRY.keys())[0] if player1 == "llm" else None
    default_model2 = list(LLM_REGISTRY.keys())[0] if player2 == "llm" else None

    return player1_model_visible, player2_model_visible, default_model1, default_model2


def update_results_tracker(scores, player1, player2):
    """Update the matrix results tracker with game outcomes."""
    player1_name = player1 if player1 != "llm" else "Human"
    player2_name = player2 if player2 != "llm" else "Human"

    # Update games played
    results_tracker[player1_name][player2_name]["games"] += 1
    results_tracker[player2_name][player1_name]["games"] += 1

    # Update wins
    if scores[0] > scores[1]:  # Player 1 wins
        results_tracker[player1_name][player2_name]["wins"] += 1
    elif scores[1] > scores[0]:  # Player 2 wins
        results_tracker[player2_name][player1_name]["wins"] += 1

    save_results_tracker()  # Save after every update


def calculate_matrix_leaderboard():
    """Generate a matrix leaderboard table."""
    matrix = [[""] + list(results_tracker.keys())]  # Header row
    for player, opponents in results_tracker.items():
        row = [player]
        for opponent in results_tracker.keys():
            games = opponents[opponent]["games"]
            wins = opponents[opponent]["wins"]
            win_percentage = (wins / games * 100) if games > 0 else 0
            row.append(f"{win_percentage:.2f}%")
        matrix.append(row)
    return matrix


def play_turn(simulator, state, player1_type, player2_type, human_move=None, player1_model=None, player2_model=None):
    """Play a single turn of the game."""
    if state.is_terminal():
        final_scores = state.returns()
        update_results_tracker(final_scores, player1_model, player2_model)
        return f"Game Over!\nFinal Scores: {final_scores}", state

    current_player = state.current_player()
    legal_moves = state.legal_actions(current_player)
    board = str(state)

    # Human player's turn
    if (player1_type == "human" and current_player == 0) or (player2_type == "human" and current_player == 1):
        if human_move is None:
            return f"Your Turn! Current Board:\n{board}\nValid Moves: {legal_moves}", state
        try:
            human_move = int(human_move)
            if human_move not in legal_moves:
                return f"Invalid move. Legal moves are: {legal_moves}\nCurrent Board:\n{board}", state
            state.apply_action(human_move)
        except ValueError:
            return f"Invalid input. Please enter a valid move number.\nValid Moves: {legal_moves}\nCurrent Board:\n{board}", state
    else:
        # LLM or bot's turn
        action = simulator._get_action(current_player, state, legal_moves)
        state.apply_action(action)

    # Continue to the next turn
    legal_moves = state.legal_actions(state.current_player())
    board = str(state)
    return f"Next Turn! Current Board:\n{board}\nValid Moves: {legal_moves}", state


# Gradio Interface
with gr.Blocks() as interface:
    with gr.Tab("Game Arena"):
        gr.Markdown("# LLM Game Arena\nPlay against LLMs or other players in classic games!")

        with gr.Row():
            game_dropdown = gr.Dropdown(
                choices=list(GAMES_REGISTRY.keys()),
                label="Select a Game",
                 value="tic_tac_toe",  # Default to Tic-Tac-Toe
            )
        with gr.Row():
            player1_dropdown = gr.Dropdown(
                choices=["human", "random_bot", "llm"],
                label="Player 1 Type",
                value="human",  # Default to human
            )
            player2_dropdown = gr.Dropdown(
                choices=["human", "random_bot", "llm"],
                label="Player 2 Type",
                value="llm",  # Default to LLM
            )
        with gr.Row():
            player1_model_dropdown = gr.Dropdown(
                choices=list(LLM_REGISTRY.keys()),
                label="Player 1 Model",
                value=None,  # No default value if Player 1 is human
                visible=False,  # Hidden by default for a human player
            )
            player2_model_dropdown = gr.Dropdown(
                choices=list(LLM_REGISTRY.keys()),
                label="Player 2 Model",
                value=list(LLM_REGISTRY.keys())[0],  # Default to the first LLM for Player 2
                visible=True,  # Visible by default for an LLM player
            )

        with gr.Row():
            human_input = gr.Textbox(label="Enter your move (number)", visible=True)
        with gr.Row():
            result_output = gr.Textbox(label="Game Progress", interactive=False)
        with gr.Row():
            restart_button = gr.Button("Restart Game")
            next_turn_button = gr.Button("Next Turn")

        # State management
        simulator_state = gr.State(None)  # To store the simulator
        game_state = gr.State(None)  # To store the game state

        restart_button.click(
            initialize_game,
            inputs=[game_dropdown, player1_dropdown, player2_dropdown, player1_model_dropdown, player2_model_dropdown],
            outputs=[simulator_state, game_state, result_output],
        )

        next_turn_button.click(
            play_turn,
            inputs=[simulator_state, game_state, player1_dropdown, player2_dropdown, human_input, player1_model_dropdown, player2_model_dropdown],
            outputs=[result_output, game_state],
        )

    with gr.Tab("Leaderboard"):
        gr.Markdown("# Matrix Leaderboard\nSee how players perform against each other!")
        leaderboard_table = gr.Dataframe(label="Leaderboard Matrix")

        def update_leaderboard_tab():
            return calculate_matrix_leaderboard()

        update_leaderboard_button = gr.Button("Update Leaderboard")
        update_leaderboard_button.click(
            update_leaderboard_tab,
            inputs=[],
            outputs=leaderboard_table,
        )

interface.launch()
