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

def merge_sqlite_logs(log_dir: str = "results/") -> pd.DataFrame:
    """
    Merges all SQLite log files in the specified directory into a single DataFrame.

    Args:
        log_dir (str): Directory where agent-specific SQLite logs are stored.

    Returns:
        pd.DataFrame: Merged DataFrame containing all moves, rewards, and game outcomes.
    """
    all_moves = []
    all_results = []

    sqlite_files = glob.glob(os.path.join(log_dir, "*.db")) # I don't understand this line

    for db_file in sqlite_files:
        agent_name = os.path.basename(db_file).replace(".db", "") # I don't understand this line

        conn = sqlite3.connect(db_file)

        # Retrieve move logs
        try:
            df_moves = pd.read_sql_query(  # I don't understand this line
                "SELECT game_name, episode, turn, action, reasoning, generation_time, opponent FROM moves",
                conn
            )
            df_moves["agent_name"] = agent_name
            all_moves.append(df_moves.drop_duplicates())  # Remove duplicates
        except Exception as e:
            print(f"No moves table in {db_file}: {e}")

        # Retrieve game results (includes rewards)
        try:
            df_results = pd.read_sql_query(
                "SELECT game_name, episode, status, reward FROM game_results", 
                conn
            )
            df_results["agent_name"] = agent_name
            all_results.append(df_results.drop_duplicates())  # Remove duplicates
        except Exception as e:
            print(f"No game_results table in {db_file}: {e}")

        conn.close()

    df_moves = pd.concat(all_moves, ignore_index=True) if all_moves else pd.DataFrame() # I don't understand this line
    df_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame() # I don't understand this line

    # Convert `opponent` lists into hashable strings before merging
    if "opponent" in df_moves.columns:
        df_moves["opponent"] = df_moves["opponent"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x) # I don't understand this line

    # Merge moves with game results (which includes rewards)
    if not df_results.empty:
        df_full = df_moves.merge(df_results, on=["game_name", "episode", "agent_name"], how="left") # I don't understand this line
    else:
        df_full = df_moves.copy() # I don't understand this line

    # Drop duplicates before returning (final safeguard)
    df_full = df_full.drop_duplicates() # I don't understand this line

    return df_full

def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Computes summary statistics from the merged results DataFrame.

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

        games_played = group["episode"].nunique()
        terminated_games = group[group["status"] == "terminated"].shape[0] if "status" in group else 0
        truncated_games = group[group["status"] == "truncated"].shape[0] if "status" in group else 0

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
    merged_df = merge_sqlite_logs(log_dir="results")
    if merged_df.empty:
        print("No log files found or merged.")
        return

    # Compute statistics
    summary = compute_summary_statistics(merged_df)

    # Ensure `agent_name` is the second column
    column_order = ["game_name", "agent_name"] + [
        col for col in merged_df.columns if col not in ["game_name", "agent_name"]
    ]
    merged_df = merged_df[column_order]

    # Save logs for review
    os.makedirs("results", exist_ok=True)
    merged_csv = os.path.join("results", f"merged_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv") # what is 'utcnow()'?
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
    print("Done.")
