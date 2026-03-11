"""
Shoe Recognition Database Module
Manages shoe recognition data using SQLite database stored in assets/models
"""
import sqlite3
import os
import json
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from utils.logger import get_logger



@dataclass
class ShoeRecord:
    """Database record for shoe recognition."""
    id: int
    shoe_type: str
    confidence: float
    features: str  # JSON-encoded feature vector
    image_hash: str  # Hash of the analyzed image
    timestamp: str
    safe_depth_cm: float
    warning_threshold: float
    critical_threshold: float


class ShoeDatabase:
    """
    SQLite database for storing and retrieving shoe recognition data.
    Database stored in assets/models directory.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize shoe recognition database.
        
        Args:
            db_path: Path to database file. If None, uses default location.
        """
        self.logger = get_logger("shoe_database")
        
        # Set default database path in assets/models
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'assets', 'models')
            os.makedirs(models_dir, exist_ok=True)
            db_path = os.path.join(models_dir, 'shoe_recognition.db')
        
        self.db_path = db_path
        self.connection = None
        
        # Initialize database
        self._initialize_database()
        self.logger.info(f"ShoeDatabase initialized at {self.db_path}")
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()
            
            # Create shoe_records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shoe_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shoe_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    features TEXT,
                    image_hash TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    safe_depth_cm REAL,
                    warning_threshold REAL,
                    critical_threshold REAL
                )
            ''')
            
            # Create shoe_types reference table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shoe_types (
                    type_name TEXT PRIMARY KEY,
                    height_cm REAL,
                    safe_depth_cm REAL,
                    description_en TEXT,
                    description_zh TEXT
                )
            ''')
            
            # Create shoe_templates table for feature matching
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shoe_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shoe_type TEXT NOT NULL,
                    descriptors BLOB,
                    keypoints_count INTEGER,
                    image_hash TEXT UNIQUE,
                    is_user_verified INTEGER DEFAULT 0
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_cache (
                    image_hash TEXT PRIMARY KEY,
                    shoe_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    last_used DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_shoe_type 
                ON shoe_records(shoe_type)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON shoe_records(timestamp)
            ''')
            
            self.connection.commit()
            self.logger.info("Database tables initialized successfully")
            
            # Populate shoe_types if empty
            self._populate_shoe_types_if_empty()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _populate_shoe_types_if_empty(self):
        """Populate shoe_types table with default data if empty."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM shoe_types")
        count = cursor.fetchone()[0]
        
        if count == 0:
            shoe_types_data = [
                ('sandals', 3.0, 0.5, 
                 'Sandals detected. Will warn for very shallow puddles.',
                 '已选择凉鞋，将对极浅水坑发出警告。'),
                ('low_shoes', 8.0, 2.0,
                 'Low shoes detected. Will warn for shallow to medium puddles.',
                 '已选择低帮鞋，将对浅至中等水坑发出警告。'),
                ('ankle_shoes', 12.0, 5.0,
                 'Ankle shoes detected. Will warn for medium puddles.',
                 '已选择踝靴，将对中等深度水坑发出警告。'),
                ('boots', 25.0, 15.0,
                 'Boots detected. Will warn for deep puddles only.',
                 '已选择靴子，仅对较深水坑发出警告。'),
                ('high_boots', 40.0, 25.0,
                 'High boots/rain boots detected. High water resistance.',
                 '已选择雨靴，防水性能优异。'),
            ]
            
            cursor.executemany('''
                INSERT INTO shoe_types 
                (type_name, height_cm, safe_depth_cm, description_en, description_zh)
                VALUES (?, ?, ?, ?, ?)
            ''', shoe_types_data)
            
            self.connection.commit()
            self.logger.info("Shoe types table populated with default data")
    
    def save_recognition(self, shoe_type: str, confidence: float,
                        features: Optional[np.ndarray] = None,
                        image_hash: Optional[str] = None) -> int:
        """
        Save a shoe recognition result to database.
        
        Args:
            shoe_type: Detected shoe type
            confidence: Confidence score (0-1)
            features: Optional feature vector from model
            image_hash: Optional hash of the image
            
        Returns:
            Record ID
        """
        try:
            cursor = self.connection.cursor()
            
            # Get shoe type parameters
            cursor.execute('''
                SELECT height_cm, safe_depth_cm 
                FROM shoe_types 
                WHERE type_name = ?
            ''', (shoe_type,))
            
            result = cursor.fetchone()
            if result:
                height_cm, safe_depth_cm = result
            else:
                height_cm, safe_depth_cm = 8.0, 2.0  # defaults
            
            # Calculate thresholds
            cm_per_frame_ratio = 50.0
            warning_threshold = safe_depth_cm / cm_per_frame_ratio
            critical_threshold = (safe_depth_cm * 0.5) / cm_per_frame_ratio
            
            # Encode features as JSON
            features_json = None
            if features is not None:
                features_json = json.dumps(features.tolist())
            
            # Insert record
            cursor.execute('''
                INSERT INTO shoe_records 
                (shoe_type, confidence, features, image_hash, 
                 safe_depth_cm, warning_threshold, critical_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (shoe_type, confidence, features_json, image_hash,
                  safe_depth_cm, warning_threshold, critical_threshold))
            
            record_id = cursor.lastrowid
            self.connection.commit()
            
            # Update cache if we have image hash
            if image_hash:
                self._update_cache(image_hash, shoe_type, confidence)
            
            self.logger.info(f"Saved recognition: {shoe_type} (confidence={confidence:.2f})")
            return record_id
            
        except Exception as e:
            self.logger.error(f"Failed to save recognition: {e}")
            return -1
    
    def save_template(self, shoe_type: str, descriptors: np.ndarray, image_hash: str, is_user_verified: int = 0):
        """Save a feature template for future matching."""
        try:
            cursor = self.connection.cursor()
            # Convert descriptors to blob
            desc_blob = sqlite3.Binary(descriptors.tobytes())
            
            cursor.execute('''
                INSERT OR REPLACE INTO shoe_templates 
                (shoe_type, descriptors, keypoints_count, image_hash, is_user_verified)
                VALUES (?, ?, ?, ?, ?)
            ''', (shoe_type, desc_blob, len(descriptors), image_hash, is_user_verified))
            self.connection.commit()
            self.logger.info(f"Saved template for {shoe_type}")
        except Exception as e:
            self.logger.error(f"Failed to save template: {e}")

    def get_all_templates(self) -> List[Tuple[str, np.ndarray]]:
        """Retrieve all shoe templates for matching."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT shoe_type, descriptors FROM shoe_templates")
            results = []
            for shoe_type, desc_blob in cursor.fetchall():
                # Convert blob back to numpy array (ORB descriptors are uint8)
                descriptors = np.frombuffer(desc_blob, dtype=np.uint8).reshape(-1, 32)
                results.append((shoe_type, descriptors))
            return results
        except Exception as e:
            self.logger.error(f"Failed to get templates: {e}")
            return []

    def get_from_cache(self, image_hash: str) -> Optional[Tuple[str, float]]:
        """
        Retrieve cached recognition result by image hash.
        
        Args:
            image_hash: Hash of the image
            
        Returns:
            Tuple of (shoe_type, confidence) or None
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT shoe_type, confidence 
                FROM recognition_cache 
                WHERE image_hash = ?
            ''', (image_hash,))
            
            result = cursor.fetchone()
            if result:
                # Update last_used timestamp
                cursor.execute('''
                    UPDATE recognition_cache 
                    SET last_used = CURRENT_TIMESTAMP 
                    WHERE image_hash = ?
                ''', (image_hash,))
                self.connection.commit()
                
                self.logger.debug(f"Cache hit for {image_hash}")
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to query cache: {e}")
            return None
    
    def _update_cache(self, image_hash: str, shoe_type: str, confidence: float):
        """Update recognition cache."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO recognition_cache 
                (image_hash, shoe_type, confidence, last_used)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (image_hash, shoe_type, confidence))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to update cache: {e}")
    
    def get_recent_recognitions(self, limit: int = 10) -> List[ShoeRecord]:
        """
        Get recent shoe recognition records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of ShoeRecord objects
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT id, shoe_type, confidence, features, image_hash, 
                       timestamp, safe_depth_cm, warning_threshold, critical_threshold
                FROM shoe_records 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            records = []
            for row in cursor.fetchall():
                records.append(ShoeRecord(*row))
            
            return records
            
        except Exception as e:
            self.logger.error(f"Failed to query recent recognitions: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get recognition statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            cursor = self.connection.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM shoe_records")
            total = cursor.fetchone()[0]
            
            # Count by type
            cursor.execute('''
                SELECT shoe_type, COUNT(*) 
                FROM shoe_records 
                GROUP BY shoe_type
            ''')
            
            stats = {'total': total}
            for shoe_type, count in cursor.fetchall():
                stats[shoe_type] = count
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {'total': 0}
    
    def clear_old_records(self, days: int = 30):
        """
        Delete records older than specified days.
        
        Args:
            days: Delete records older than this many days
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                DELETE FROM shoe_records 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            ''', (days,))
            
            deleted = cursor.rowcount
            self.connection.commit()
            
            self.logger.info(f"Deleted {deleted} records older than {days} days")
            
        except Exception as e:
            self.logger.error(f"Failed to clear old records: {e}")
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
