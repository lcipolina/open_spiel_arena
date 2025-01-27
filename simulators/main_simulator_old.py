"""
main_simulation.py

Example script that demonstrates how to use the OpenSpielEnv and agent classes
to run one or more episodes of a game.
"""

from envs.open_spiel_env import OpenSpielEnv
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from enum import Enum, unique

class PlayerType(Enum):
    HUMAN = "human"
    RANDOM_BOT = "random_bot"
    LLM = "llm"
    SELF_PLAY = "self_play"

def main():
    # Load an OpenSpiel game object:
    import pyspiel
    game = pyspiel.load_game("tic_tac_toe")  # example

    # Example: define player types for Player 1 and Player 2
    player_type = {
        "Player 1": "human",     # or "llm" / "random_bot"
        "Player 2": "random_bot"
    }

    # Construct the environment
    env = OpenSpielEnv(
        game=game,
        game_name="Tic Tac Toe",
        player_type=player_type,
        max_game_rounds=None  # or set a limit
    )

    # Build the appropriate agents for each player
    # Typically, you'd store them in a list or dict keyed by player index
    agents = {
        "Player 1": HumanAgent(game_name="Tic Tac Toe"),
        "Player 2": RandomAgent()
    }

    outcomes = env._initialize_outcomes()  # Reuse environment's method

    # Let's simulate multiple rounds
    num_rounds = 2
    for _ in range(num_rounds):
        # 1) Reset environment
        obs = env.reset()

        # 2) Play until done
        done = False
        while not done:
            current_player = env.state.current_player()
            if env.normalize_player_id(current_player) < 0:
                # It's a chance or simultaneous node. This example won't handle that.
                raise NotImplementedError("Chance/simultaneous not handled in main script.")

            # Build "Player i" name
            player_name = f"Player {current_player + 1}"
            legal_actions = env.state.legal_actions(current_player)

            # Ask the correct agent for an action
            action = agents[player_name].choose_action(legal_actions, env.state)

            # Step the environment
            obs, reward, done, info = env.step(action)

            # Optional: render or log
            env.render()

        # 3) If done, record outcomes
        if "final_scores" in info:
            winner = env.record_outcomes(info["final_scores"], outcomes)
            print(f"Round finished. Winner: {winner}\n")

    # Print final aggregated outcomes
    print("Aggregated outcomes after all rounds:", outcomes)

    env.close()

if __name__ == "__main__":
    main()
