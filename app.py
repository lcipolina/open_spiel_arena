'''
Gradio app for the model
'''

import os
import json
import sqlite3
import glob
import pandas as pd
import gradio as gr
from datetime import datetime
from typing import Dict, List

# Directory to store SQLite results
db_dir = "results/"

def find_or_download_db():
    """Check if SQLite .db files exist; if not, attempt to download from cloud storage."""
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    db_files = glob.glob(os.path.join(db_dir, "*.db"))

    # Ensure the random bot database exists
    if "results/random_None.db" not in db_files:
        raise FileNotFoundError("Please upload results for the random agent in a file named 'random_None.db'.")

    return db_files

def extract_agent_info(filename: str):
    """Extract agent type and model name from the filename."""
    base_name = os.path.basename(filename).replace(".db", "")
    parts = base_name.split("_", 1)
    if len(parts) == 2:
        agent_type, model_name = parts
    else:
        agent_type, model_name = parts[0], "Unknown"
    return agent_type, model_name

def get_available_games() -> List[str]:
    """Extracts all unique game names from all SQLite databases and includes 'Aggregated Performance'."""
    db_files = find_or_download_db()
    game_names = set()

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        try:
            query = "SELECT DISTINCT game_name FROM moves"
            df = pd.read_sql_query(query, conn)
            game_names.update(df["game_name"].tolist())
        except Exception:
            pass  # Ignore errors if table doesn't exist
        finally:
            conn.close()

    game_list = sorted(game_names) if game_names else ["No Games Found"]
    game_list.insert(0, "Aggregated Performance")  # Ensure 'Aggregated Performance' is always first
    return game_list

def extract_leaderboard_stats(game_name: str) -> pd.DataFrame:
    """Extract and aggregate leaderboard stats from all SQLite databases."""
    db_files = find_or_download_db()
    all_stats = []

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        agent_type, model_name = extract_agent_info(db_file)

        # Skip random agent rows
        if agent_type == "random":
            conn.close()
            continue

        if game_name == "Aggregated Performance":
            query = "SELECT COUNT(DISTINCT episode) AS games_played, " \
                    "SUM(reward) AS total_rewards " \
                    "FROM game_results"
            df = pd.read_sql_query(query, conn)

            # Use avg_generation_time from a specific game (e.g., Kuhn Poker)
            game_query = "SELECT AVG(generation_time) FROM moves WHERE game_name = 'kuhn_poker'"
            avg_gen_time = conn.execute(game_query).fetchone()[0] or 0
        else:
            query = "SELECT COUNT(DISTINCT episode) AS games_played, " \
                    "SUM(reward) AS total_rewards " \
                    "FROM game_results WHERE game_name = ?"
            df = pd.read_sql_query(query, conn, params=(game_name,))

            # Fetch average generation time from moves table
            gen_time_query = "SELECT AVG(generation_time) FROM moves WHERE game_name = ?"
            avg_gen_time = conn.execute(gen_time_query, (game_name,)).fetchone()[0] or 0

        # Keep division by 2 for total rewards
        df["total_rewards"] = df["total_rewards"].fillna(0).astype(float) / 2

        # Ensure avg_gen_time has decimals
        avg_gen_time = round(avg_gen_time, 3)

        # Calculate win rate against random bot using moves table
        vs_random_query = """
            SELECT COUNT(DISTINCT gr.episode) FROM game_results gr
            JOIN moves m ON gr.game_name = m.game_name AND gr.episode = m.episode
            WHERE m.opponent = 'random_None' AND gr.reward > 0
        """
        total_vs_random_query = """
            SELECT COUNT(DISTINCT gr.episode) FROM game_results gr
            JOIN moves m ON gr.game_name = m.game_name AND gr.episode = m.episode
            WHERE m.opponent = 'random_None'
        """
        wins_vs_random = conn.execute(vs_random_query).fetchone()[0] or 0
        total_vs_random = conn.execute(total_vs_random_query).fetchone()[0] or 0
        vs_random_rate = (wins_vs_random / total_vs_random * 100) if total_vs_random > 0 else 0

        df.insert(0, "agent_name", model_name)  # Ensure agent_name is the first column
        df.insert(1, "agent_type", agent_type)  # Ensure agent_type is second column
        df["avg_generation_time (sec)"] = avg_gen_time
        df["win vs_random (%)"] = round(vs_random_rate, 2)

        all_stats.append(df)
        conn.close()

    leaderboard_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    if leaderboard_df.empty:
        leaderboard_df = pd.DataFrame(columns=["agent_name", "agent_type", "# games", "total rewards", "avg_generation_time (sec)", "win-rate", "win vs_random (%)"])

    return leaderboard_df

def generate_leaderboard_json():
    """Generate a JSON file containing leaderboard stats."""
    available_games = get_available_games()
    leaderboard = extract_leaderboard_stats("Aggregated Performance").to_dict(orient="records")
    json_file = "results/leaderboard_stats.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat(), "leaderboard": leaderboard}, f, indent=4)
    return json_file

with gr.Blocks() as interface:
    with gr.Tab("Leaderboard"):
        gr.Markdown("# LLM Model Leaderboard\nTrack performance across different games!")
        available_games = get_available_games()
        leaderboard_game_dropdown = gr.Dropdown(available_games, label="Select Game", value="Aggregated Performance")
        leaderboard_table = gr.Dataframe(headers=["agent_name", "agent_type", "# games", "total rewards", "avg_generation_time (sec)", "win-rate", "win vs_random (%)"])
        refresh_button = gr.Button("Refresh Leaderboard")
        generate_button = gr.Button("Generate Leaderboard JSON")
        download_component = gr.File(label="Download Leaderboard JSON")

        leaderboard_game_dropdown.change(extract_leaderboard_stats, inputs=[leaderboard_game_dropdown], outputs=[leaderboard_table])
        refresh_button.click(extract_leaderboard_stats, inputs=[leaderboard_game_dropdown], outputs=[leaderboard_table])
        generate_button.click(generate_leaderboard_json, outputs=[download_component])

interface.launch()
