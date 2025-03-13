import sqlite3
import os
from datetime import datetime
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
        self.db_path = f"results/{agent_type}_{model_name.replace('-', '_')}.db"
        os.makedirs("results", exist_ok=True)
        self._create_database()

    def _create_database(self):
        """
        Creates necessary tables if they do not exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create 'moves' table (logs every move taken in a game)
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

        # Create 'rewards' table (stores final rewards for each episode)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                reward REAL,
                timestamp TEXT
            )
        """)

        # Create 'illegal_moves' table (records illegal moves made during the game)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS illegal_moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_name TEXT,
                episode INTEGER,
                illegal_action INTEGER,
                timestamp TEXT
            )
        """)

        # Create 'game_results' table (stores final results of games played)
        # TODO: add opponents to this table !
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

        conn.commit()
        conn.close()

    def log_game_result(self, game_name: str, episode: int, status: str, reward: float):
        """
        Logs the final result of a game.

        Args:
            game_name (str): Name of the game played.
            episode (int): Episode number.
            status (str): 'terminated' or 'truncated'.
            reward (float): Final reward the agent received.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO game_results (game_name, episode, status, reward, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            game_name, episode, status, reward, datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def log_move(self, game_name: str, episode: int, turn: int, action: int,
                 reasoning: str, opponent: str, generation_time: float,
                 agent_type: str, agent_model: str):
        """
        Logs an agent's move into the database.

        Args:
            game_name (str): Name of the game.
            episode (int): Episode number.
            turn (int): Turn number in the episode.
            action (int): The action taken by the agent.
            reasoning (str): The agent's reasoning for the action.
            opponent (str): The opponent's identifier.
            generation_time (float): Time taken to generate the move.
            agent_type (str): Type of agent (e.g., "llm", "random", "human").
            agent_model (str): Specific model name for LLMs.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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
        """
        Logs the final reward for an agent after an episode.

        Args:
            game_name (str): Name of the game.
            episode (int): Episode number.
            reward (float): Final reward received.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO rewards (game_name, episode, reward, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            game_name, episode, reward, datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def get_agent_moves(self):
        """
        Retrieves all actions taken by the agent.

        Returns:
            List of tuples containing game_name, episode, turn, action,
            reasoning, opponent, generation_time, and timestamp.
        """
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
        """
        Retrieves all rewards logged for the agent.

        Returns:
            List of tuples containing game_name, episode, reward, and timestamp.
        """
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

    def get_game_results(self, game_name: str):
        """
        Retrieves all game results for a specific game.

        Args:
            game_name (str): Name of the game to fetch results for.

        Returns:
            List of tuples containing game_name, episode, status, reward, and timestamp.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT game_name, episode, status, reward, timestamp
            FROM game_results
            WHERE game_name = ?
            ORDER BY episode
        """, (game_name,))

        game_results = cursor.fetchall()
        conn.close()
        return game_results

    def get_agent_results(self):
        """
        Fetches all game results for the agent's database.

        Returns:
            Pandas DataFrame containing game_name, episode, status, reward, and timestamp.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT game_name, episode, status, reward, timestamp
            FROM game_results
            ORDER BY game_name, episode
        """, conn)
        conn.close()
        return df
