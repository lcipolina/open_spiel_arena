import gradio as gr
from simulators.tic_tac_toe_simulator import TicTacToeSimulator
from simulators.prisoners_dilemma_simulator import PrisonersDilemmaSimulator
from simulators.rock_paper_scissors_simulator import RockPaperScissorsSimulator
from games_registry import GAMES_REGISTRY
from simulators.base_simulator import PlayerType
from typing import Dict


# Initialize leaderboard
leaderboard = {}


def play_game(game_name, player1_type, player2_type, player1_model, player2_model, rounds):
    """Play the selected game with specified players."""
    # Get game and simulator
    game_config = GAMES_REGISTRY[game_name]
    game = game_config["loader"]()
    simulator_class = game_config["simulator"]

    # Initialize the simulator
    llms = {}
    if player1_type == "llm":
        llms["Player 1"] = player1_model
    if player2_type == "llm":
        llms["Player 2"] = player2_model

    simulator = simulator_class(game, game_name, llms=llms)

    # Collect game states for display
    game_states = []

    def log_fn(state):
        """Log current state and legal moves."""
        current_player = state.current_player()
        legal_moves = state.legal_actions(current_player)
        board = str(state)
        game_states.append(f"Current Player: {current_player}\nBoard:\n{board}\nLegal Moves: {legal_moves}")

    # Play the game with rounds
    results = simulator.simulate(rounds=int(rounds), log_fn=log_fn)

    # Display game states and results
    game_states_str = "\n\n".join(game_states)
    leaderboard_str = "\n".join([f"{name}: {score} wins" for name, score in leaderboard.items()])
    return f"Game States:\n{game_states_str}\n\nResults for {game_name}:\n{results}\n\nLeaderboard:\n{leaderboard_str}"


def toggle_model_dropdown(player1, player2):
    """Control visibility of model dropdowns based on player types."""
    player1_model_visible = gr.update(visible=(player1 == "llm"))
    player2_model_visible = gr.update(visible=(player2 == "llm"))
    return player1_model_visible, player2_model_visible


with gr.Blocks() as interface:
    gr.Markdown("# LLM Game Arena\nSelect a game and players to play against LLMs.")

    with gr.Row():
        game_dropdown = gr.Dropdown(
            choices=list(GAMES_REGISTRY.keys()),
            label="Select a Game",
            value=list(GAMES_REGISTRY.keys())[0],  # Default to the first game
        )
    with gr.Row():
        player1_dropdown = gr.Dropdown(
            choices=["human", "random_bot", "llm"],
            label="Player 1 Type",
            value="llm",  # Default to LLM
        )
        player2_dropdown = gr.Dropdown(
            choices=["human", "random_bot", "llm"],
            label="Player 2 Type",
            value="random_bot",  # Default to Random Bot
        )
    with gr.Row():
        player1_model_dropdown = gr.Dropdown(
            choices=["gpt2", "google/flan-t5-small"],
            label="Player 1 Model",
            visible=False,
        )
        player2_model_dropdown = gr.Dropdown(
            choices=["gpt2", "google/flan-t5-small"],
            label="Player 2 Model",
            visible=False,
        )
    with gr.Row():
        rounds_slider = gr.Slider(1, 10, step=1, label="Number of Rounds", value=3)  # Default to 3 rounds
    with gr.Row():
        result_output = gr.Textbox(label="Game Results", interactive=False)

    # Event triggers for dynamic updates
    player1_dropdown.change(
        toggle_model_dropdown,
        inputs=[player1_dropdown, player2_dropdown],
        outputs=[player1_model_dropdown, player2_model_dropdown],
    )
    player2_dropdown.change(
        toggle_model_dropdown,
        inputs=[player1_dropdown, player2_dropdown],
        outputs=[player1_model_dropdown, player2_model_dropdown],
    )

    # Button to play the game
    play_button = gr.Button("Play Game")
    play_button.click(
        play_game,
        inputs=[
            game_dropdown,
            player1_dropdown,
            player2_dropdown,
            player1_model_dropdown,
            player2_model_dropdown,
            rounds_slider,
        ],
        outputs=result_output,
    )

interface.launch()
