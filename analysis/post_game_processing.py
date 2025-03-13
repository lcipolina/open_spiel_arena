#!/usr/bin/env python3
"""
post_match_processing.py

Merges all agent-specific SQLite logs, computes summary statistics,
and stores results in 'results/'.
"""

import sqlite3
import glob
import os
import json
import pandas as pd
from datetime import datetime

def merge_sqlite_logs(log_dir: str = "logs/") -> pd.DataFrame:
    """
    Merges all SQLite log files in the specified directory into a single DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame containing all moves, rewards, and game outcomes.
    """
    all_moves = []
    all_rewards = []
    all_results = []

    sqlite_files = glob.glob(os.path.join(log_dir, "*.db"))

    for db_file in sqlite_files:
        base_name = os.path.basename(db_file).replace(".db", "")

        try:
            agent_name, game_name, _ = base_name.split("_")
        except ValueError:
            print(f"Skipping file with unexpected format: {db_file}")
            continue

        conn = sqlite3.connect(db_file)

        # Load moves table
        try:
            df_moves = pd.read_sql_query("SELECT episode, turn, action, reasoning, generation_time, opponent FROM moves", conn)
            df_moves["agent_name"] = agent_name
            df_moves["game_name"] = game_name

            # ✅ Convert `opponent` from a comma-separated string back into a list
            df_moves["opponent"] = df_moves["opponent"].apply(lambda x: x.split(", ") if isinstance(x, str) else [])

            all_moves.append(df_moves)
        except Exception as e:
            print(f"No moves table in {db_file}: {e}")

        # Load rewards table
        try:
            df_rewards = pd.read_sql_query("SELECT episode, reward FROM rewards", conn)
            df_rewards["agent_name"] = agent_name
            df_rewards["game_name"] = game_name
            all_rewards.append(df_rewards)
        except Exception as e:
            print(f"No rewards table in {db_file}: {e}")

        conn.close()

    # Merge all data into one DataFrame
    df_moves = pd.concat(all_moves, ignore_index=True) if all_moves else pd.DataFrame()
    df_rewards = pd.concat(all_rewards, ignore_index=True) if all_rewards else pd.DataFrame()

    # Load game results (global results, not per agent)
    try:
        global_db = sqlite3.connect("logs/game_logs_global.db")
        df_results = pd.read_sql_query("SELECT game_name, episode, status FROM game_results", global_db)
        global_db.close()
        all_results.append(df_results)
    except Exception as e:
        print(f"No global game_results table found: {e}")

    df_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    # Merge everything together
    if not df_moves.empty and not df_rewards.empty:
        df_full = df_moves.merge(df_rewards, on=["game_name", "episode", "agent_name"], how="left")
    else:
        df_full = df_moves if not df_moves.empty else df_rewards

    if not df_results.empty:
        df_full = df_full.merge(df_results, on=["game_name", "episode"], how="left")

    return df_full

def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Computes summary statistics from the merged log DataFrame.

    Returns:
        dict: Summary statistics keyed by game and agent.
    """
    summary = {}
    if df.empty:
        return summary

    for (game, agent), group in df.groupby(["game_name", "agent_name"]):
        total_moves = group.shape[0]
        avg_gen_time = group["generation_time"].mean() if not group["generation_time"].empty else None
        total_rewards = group["reward"].sum() if "reward" in group else None

        # Compute win/loss/draw statistics
        games_played = group["episode"].nunique()
        if "status" in group:
            terminated_games = group[group["status"] == "terminated"].shape[0]
            truncated_games = group[group["status"] == "truncated"].shape[0]
        else:
            terminated_games, truncated_games = None, None

        summary.setdefault(game, {})[agent] = {
            "games_played": games_played,
            "total_moves": total_moves,
            "average_generation_time": avg_gen_time,
            "total_rewards": total_rewards,
            "games_terminated": terminated_games,
            "games_truncated": truncated_games
        }
    return summary


def save_summary(summary: dict, output_dir: str = "results/") -> str:
    """
    Saves the summary statistics to a JSON file.

    Returns:
        str: The path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"merged_game_results_{timestamp}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {file_path}")
    return file_path

###########################################################
# Main entry point
###########################################################
def main():
    #  Merge all SQLite logs
    merged_df = merge_sqlite_logs(log_dir="logs")
    if merged_df.empty:
        print("No log files found or merged.")
        return

    # Compute statistics
    summary = compute_summary_statistics(merged_df)

    # Save logs for review
    os.makedirs("results", exist_ok=True)
    merged_csv = os.path.join("results", f"merged_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
    merged_df.to_csv(merged_csv, index=False)
    print(f"Merged logs saved as CSV to {merged_csv}")

    # Save summary results
    save_summary(summary, output_dir="results")

    # Show how games ended
    print("Game Outcomes Summary:")
    if "status" in merged_df:
        game_end_counts = merged_df["status"].value_counts()
        print(game_end_counts)

    # Display first 5 moves
    print("\nMerged Log DataFrame (First 5 Rows):")
    print(merged_df.head())

if __name__ == "__main__":
    main()
