import sqlite3
import os, json
from datetime import datetime
from typing import Dict
import pandas as pd

class SQLiteLogger:
    """Handles logging of agent decisions into a structured SQLite database."""

    def __init__(self, agent_type: str, model_name: str):
        """
        Initializes the logger with a unique database file for each agent type and model.

        Args:
            agent_type (str): The type of agent (e.g., "llm", "random", "human").
            model_name (str): The model name (for LLMs), or "None" for random/human.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.db_path = f"results/{agent_type}_{model_name.replace('-', '_')}.db"
        os.makedirs("results", exist_ok=True)

        self._create_database()

    def _create_database(self):
        """Creates tables for moves, rewards, illegal moves, and game results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for actions (logged per turn)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                turn INTEGER,
                action INTEGER,
                reasoning TEXT,
                opponent TEXT,
                generation_time REAL,
                agent_type TEXT,  -- Store whether agent was LLM, random, or human
                agent_model TEXT, -- Store the specific model name (for LLMs)
                timestamp TEXT
            )
        """)

        # Table for rewards (logged per episode)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                reward REAL,
                timestamp TEXT
            )
        """)

        # Table for illegal moves (logged per episode)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS illegal_moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                illegal_action INTEGER,
                timestamp TEXT
            )
        """)

        # Table for game results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                status TEXT,  -- 'terminated' or 'truncated'
                reward REAL,  -- Final reward the agent received
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log_game_result(self, game_name: str, episode: int, status: str, reward: float):
        """
        Logs final game results per agent into their own SQLite database.

        Args:
            game_name: Name of the game played.
            episode: Episode number.
            status: 'terminated' or 'truncated'.
            reward: Final reward the agent received.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                status TEXT,
                reward REAL,
                timestamp TEXT
            )
        """)

        # Insert the game result
        cursor.execute("""
            INSERT INTO game_results (game_name, episode, status, reward, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            game_name,
            episode,
            status,
            reward,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def log_move(self, game_name: str, episode: int, turn: int, action: int,
             reasoning: str, opponent: str, generation_time: float,
             agent_type: str, agent_model: str):
        """
        Logs an agent's move into the SQLite database.
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                turn INTEGER,
                action INTEGER,
                reasoning TEXT,
                opponent TEXT,
                generation_time REAL,
                agent_type TEXT,
                agent_model TEXT,
                timestamp TEXT
            )
        """)

        cursor.execute("""
            INSERT INTO moves (game_name, episode, turn, action, reasoning,
                            opponent, generation_time, agent_type, agent_model, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_name, episode, turn, action, reasoning,
            opponent, generation_time, agent_type, agent_model, datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def log_rewards(self, game_name: str, episode: int, reward: float):
        """Logs the final reward for an agent after an episode."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO rewards (game_name, episode, reward, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            game_name,
            episode,
            reward,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

    def get_agent_moves(self):
        """Retrieves all actions taken by the agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT game_name, episode, turn, action, reasoning, opponent, generation_time, timestamp
            FROM moves
            ORDER BY game_name, episode, turn
        """)
        moves = cursor.fetchall()
        conn.close()
        return moves

    def get_agent_rewards(self):
        """Retrieves all rewards for the agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT game_name, episode, reward, timestamp
            FROM rewards
            ORDER BY game_name, episode
        """)
        rewards = cursor.fetchall()
        conn.close()
        return rewards

    def get_game_results(game_name: str):
        """Retrieves all game results for a given game."""
        conn = sqlite3.connect(f"logs/game_logs_{game_name}.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT episode, status, reward, timestamp
            FROM game_results
            WHERE game_name = ?
            ORDER BY episode
        """, (game_name,))

        game_results = cursor.fetchall()
        conn.close()
        return game_results

    def get_agent_results(agent_name: str):
        """Fetches all game results for an agent."""
        conn = sqlite3.connect(f"results/{agent_name}.db")
        df = pd.read_sql_query("""
            SELECT game_name, episode, status, reward, timestamp
            FROM game_results
            ORDER BY game_name, episode
        """, conn)
        conn.close()
        return df

#  Example: Get results for LLM agent - TODO
#df_llm = get_agent_results("llm_gpt4")
#print(df_llm.head())
