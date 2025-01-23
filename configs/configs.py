"""
configs.py

Holds various simulation or experiment configs.
"""

def default_simulation_config():
    """
    Returns a dictionary specifying how to run our simulation.
    This is analogous to an RLlib config object, but simpler.
    """
    return {
        # Which OpenSpiel game to load
        "game_name": "tic_tac_toe",

        # Number of repeated episodes to play. Used to obtain more robust outcome stats.
        "rounds": 5,

        # Maximum number of rounds for iterated games. Ignored by single-shot games.
        "max_game_rounds": 1,

        # Agents: a list or dict describing each player
        # 'type' can be "human", "random_bot", "llm", etc.
        # 'name' is optional (e.g., "Player 1")
        "agents": [
            {"type": "human", "name": "Player 1"},
            {"type": "random_bot", "name": "Player 2"}
        ],

        # Random seed
        "seed": 42,

        # If you want to alternate which agent goes first
        "alternate_first_player": True,

        # Optionally pass other settings:
        # "llm_model_names": [],
        # "max_game_rounds": None,
    }
