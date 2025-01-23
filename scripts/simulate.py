#!/usr/bin/env python3
"""
simulate.py

Runs a game simulation using a config dict, similar to how RLlib passes
arguments around. This script can:
- Load an OpenSpiel game from `config["game_name"]`
- Construct environment
- Construct agents
- Play multiple rounds, optionally seeding RNG, alternating first player, etc.
"""

import random

# Suppose your env is in envs.open_spiel_env
from envs.open_spiel_env import OpenSpielEnv

# Agents:
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
# from agents.trained_agent import TrainedAgent #TODO (lck) implement this

# Game-play config
from configs import default_simulation_config

import pyspiel

def run_simulation(config):
    """
    Runs the OpenSpiel simulation given a config dictionary.
    Returns a dictionary with aggregated outcomes or stats.

    Config keys may include:
      - game_name (str): e.g. "tic_tac_toe"
      - rounds (int): how many episodes to play
      - agents (list of dict): each dict has "type": "human"/"llm"/"random_bot", etc.
      - seed (int or None)
      - alternate_first_player (bool)
      - max_game_rounds (int or None): for iterated games.
      ...
    """

    # 1. Set up random seed if specified
    if config.get("seed") is not None:
        random.seed(config["seed"])

    # 2. Load the game via pyspiel
    game_name = config["game_name"]
    game = pyspiel.load_game(game_name)

    # 3. Build player_type map, e.g. {"Player 1": "human", "Player 2": "random_bot"}
    #    from the config's "agents" list
    # If an agent doesn't specify "name", we'll assign "Player i+1"
    agents_info = config["agents"]
    player_type_map = {}
    llms = {}  # if needed

    for i, agent_cfg in enumerate(agents_info):
        # agent_cfg might look like: {"type": "human", "name": "Player 1"}
        name = agent_cfg.get("name", f"Player {i+1}")
        player_type_map[name] = agent_cfg["type"]

        # If LLM needed:
        # if agent_cfg["type"] == "llm":
        #    llms[name] = some_llm_loader(...)

    # 4. Create the environment
    max_game_rounds = config.get("max_game_rounds", 1)
    env = OpenSpielEnv(
        game=game,
        game_name=game_name,
        player_type=player_type_map,
        llms=llms,
        max_game_rounds=max_game_rounds
    )

    # 5. Build actual agent instances for each player (human, random, LLM, etc.)
    agents_dict = {}
    # We might store them in the same order as "Player 1", "Player 2", etc.
    player_names = list(player_type_map.keys())

    for p_name in player_names:
        agent_type = player_type_map[p_name]
        if agent_type == "human":
            agents_dict[p_name] = HumanAgent(env.game_name)
        elif agent_type == "random_bot":
            agents_dict[p_name] = RandomAgent()
        # elif agent_type == "llm":
        #     llm_instance = llms[p_name]
        #     agents_dict[p_name] = LLMAgent(llm_instance, env.game_name)
        # elif agent_type == "trained":
        #     agents_dict[p_name] = TrainedAgent("checkpoint.path")
        else:
            print(f"Unrecognized agent type '{agent_type}'. Defaulting to random bot.")
            agents_dict[p_name] = RandomAgent()

    # 6. Main simulation loop
    # We'll accumulate outcomes in env._initialize_outcomes()
    outcomes = env._initialize_outcomes()
    rounds = config["rounds"]
    alternate_first = config.get("alternate_first_player", False)

    for episode_idx in range(rounds):
        # If you want to vary seeds each round:
        seed_base = config.get("seed")
        if seed_base is not None:
            round_seed = seed_base + episode_idx
            random.seed(round_seed)

        # Possibly reorder the agent dictionary if we want to alternate who goes first
        if alternate_first and episode_idx % 2 == 1 and len(player_names) == 2:
            # swap the mapping for Player 1 and Player 2
            # This logic depends on how your environment identifies "current_player".
            # Another approach is to set the environment's starting player manually.
            pass  # For simplicity, not fully implemented here

        obs = env.reset()
        done = False

        print(f"\n=== Starting Round {episode_idx + 1} ===")
        while not done:
            current_player = env.state.current_player()
            normalized_id = env.normalize_player_id(current_player)

            if normalized_id < 0:
                # handle chance/simultaneous/terminal
                if normalized_id == -1:  # CHANCE
                    env._handle_chance_node(env.state)
                    continue
                elif normalized_id == -2:  # SIMULTANEOUS
                    # gather actions from each player
                    actions = env._collect_actions(env.state)
                    env.state.apply_actions(actions)
                    continue
                elif normalized_id == -4:  # TERMINAL
                    break
                else:
                    raise ValueError(f"Unexpected special player ID: {normalized_id}")

            # Normal turn-based
            player_name = player_names[current_player]  # e.g. "Player 1"
            agent = agents_dict[player_name]
            legal_actions = env.state.legal_actions(current_player)

            action = agent.choose_action(legal_actions, env.state)
            obs, reward, done, info = env.step(action)
            env.render()

        # Record final outcomes if done
        if "final_scores" in info:
            final_scores = info["final_scores"]
            winner = env.record_outcomes(final_scores, outcomes)
            print(f"Round {episode_idx+1} finished. Winner: {winner}")

    env.close()
    return outcomes


def main():
    """
    Calls run_simulation with a config, then prints results.
    """

    config = default_simulation_config()
    #  TODO (lck) parse JSON or a CLI argument to pick another config.

    results = run_simulation(config)
    print("\nFinal aggregated outcomes:", results)


if __name__ == "__main__":
    main()
