import gradio as gr
import random

# Function to simulate a game
def play_game(game, player1, player2):
    if game == "tic_tac_toe":
        # Example game logic (placeholder)
        winner = random.choice([player1, player2, "Draw"])
        return f"The winner is: {winner}"
    else:
        return "Game logic not implemented yet."

# Create the UI
with gr.Blocks() as demo:
    gr.Markdown("# LLM Arena Demo")

    game = gr.Dropdown(["tic_tac_toe", "rock_paper_scissors"], label="Select a Game")
    player1 = gr.Dropdown(["human", "LLM", "random_bot"], label="Player 1")
    player2 = gr.Dropdown(["human", "LLM", "random_bot"], label="Player 2")

    play_button = gr.Button("Play")
    result = gr.Textbox(label="Game Result")

    play_button.click(play_game, inputs=[game, player1, player2], outputs=result)

# Launch the demo
demo.launch()
