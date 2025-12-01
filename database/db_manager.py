"""Database manager for user sessions and history"""
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json


class DatabaseManager:
    def __init__(self, db_path: str = "drug_recommendation.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Search history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                condition TEXT NOT NULL,
                recommendations TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str) -> Optional[int]:
        """Create a new user and return user_id"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            return None
    
    def get_user_id(self, username: str) -> Optional[int]:
        """Get user_id by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def add_search_history(self, user_id: int, condition: str, recommendations: List[Dict]):
        """Add search to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO search_history (user_id, condition, recommendations) VALUES (?, ?, ?)",
            (user_id, condition, json.dumps(recommendations))
        )
        conn.commit()
        conn.close()
    
    def get_user_history(self, user_id: int) -> List[Dict]:
        """Get user's search history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT condition, recommendations, timestamp FROM search_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
            (user_id,)
        )
        results = cursor.fetchall()
        conn.close()
        
        history = []
        for condition, recommendations, timestamp in results:
            history.append({
                "condition": condition,
                "recommendations": json.loads(recommendations),
                "timestamp": timestamp
            })
        return history
    