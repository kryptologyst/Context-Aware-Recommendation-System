"""Data structures and utilities for context-aware recommendation system."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class ContextFeatures:
    """Context features for recommendations."""
    
    TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
    DAY_OF_WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    SEASONS = ["spring", "summer", "autumn", "winter"]
    WEATHER = ["sunny", "cloudy", "rainy", "snowy"]
    DEVICE_TYPES = ["mobile", "desktop", "tablet", "tv"]
    LOCATION_TYPES = ["home", "work", "outdoor", "travel", "restaurant", "gym"]


class DatasetGenerator:
    """Generate synthetic context-aware recommendation datasets."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize dataset generator.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.n_users = config.data.n_users
        self.n_items = config.data.n_items
        self.n_interactions = config.data.n_interactions
        self.min_interactions = config.data.min_interactions_per_user
        self.max_interactions = config.data.max_interactions_per_user
        
    def generate_items(self) -> pd.DataFrame:
        """Generate item metadata with context preferences.
        
        Returns:
            DataFrame with item information and context preferences.
        """
        items = []
        
        # Define item categories with context preferences
        categories = {
            "morning_coffee": {"category": "beverage", "time_pref": "morning", "location_pref": "work"},
            "breakfast_food": {"category": "food", "time_pref": "morning", "location_pref": "home"},
            "lunch_meal": {"category": "food", "time_pref": "afternoon", "location_pref": "restaurant"},
            "evening_movie": {"category": "entertainment", "time_pref": "evening", "location_pref": "home"},
            "workout_gear": {"category": "fitness", "time_pref": "afternoon", "location_pref": "gym"},
            "travel_guide": {"category": "travel", "time_pref": "any", "location_pref": "travel"},
            "night_snack": {"category": "food", "time_pref": "night", "location_pref": "home"},
            "weekend_activity": {"category": "entertainment", "time_pref": "any", "location_pref": "outdoor"},
        }
        
        for i in range(self.n_items):
            # Select category based on item index
            category_key = list(categories.keys())[i % len(categories)]
            category_info = categories[category_key]
            
            item = {
                "item_id": f"item_{i:04d}",
                "title": f"{category_key.replace('_', ' ').title()} {i}",
                "category": category_info["category"],
                "time_preference": category_info["time_pref"],
                "location_preference": category_info["location_pref"],
                "price": np.random.uniform(5, 100),
                "rating": np.random.uniform(3.0, 5.0),
                "popularity": np.random.exponential(1.0),
            }
            items.append(item)
            
        return pd.DataFrame(items)
    
    def generate_users(self) -> pd.DataFrame:
        """Generate user profiles with context preferences.
        
        Returns:
            DataFrame with user information and context preferences.
        """
        users = []
        
        for i in range(self.n_users):
            user = {
                "user_id": f"user_{i:04d}",
                "age": np.random.randint(18, 65),
                "gender": np.random.choice(["male", "female", "other"]),
                "preferred_time": np.random.choice(ContextFeatures.TIME_OF_DAY),
                "preferred_location": np.random.choice(ContextFeatures.LOCATION_TYPES),
                "activity_level": np.random.choice(["low", "medium", "high"]),
                "budget_level": np.random.choice(["low", "medium", "high"]),
            }
            users.append(user)
            
        return pd.DataFrame(users)
    
    def generate_interactions(self, items_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        """Generate user-item interactions with context.
        
        Args:
            items_df: Items DataFrame.
            users_df: Users DataFrame.
            
        Returns:
            DataFrame with interactions and context information.
        """
        interactions = []
        
        # Generate interactions for each user
        for _, user in users_df.iterrows():
            n_user_interactions = np.random.randint(
                self.min_interactions, 
                min(self.max_interactions + 1, self.n_items)
            )
            
            # Sample items for this user
            user_items = np.random.choice(
                items_df["item_id"].values, 
                size=n_user_interactions, 
                replace=False
            )
            
            for item_id in user_items:
                # Get item info
                item_info = items_df[items_df["item_id"] == item_id].iloc[0]
                
                # Generate context
                time_of_day = np.random.choice(ContextFeatures.TIME_OF_DAY)
                day_of_week = np.random.choice(ContextFeatures.DAY_OF_WEEK)
                season = np.random.choice(ContextFeatures.SEASONS)
                weather = np.random.choice(ContextFeatures.WEATHER)
                device_type = np.random.choice(ContextFeatures.DEVICE_TYPES)
                location_type = np.random.choice(ContextFeatures.LOCATION_TYPES)
                
                # Generate timestamp (last 6 months)
                days_ago = np.random.randint(0, 180)
                hours_ago = np.random.randint(0, 24)
                timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
                
                # Calculate interaction weight based on context match
                weight = self._calculate_context_weight(
                    user, item_info, time_of_day, location_type
                )
                
                interaction = {
                    "user_id": user["user_id"],
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "weight": weight,
                    "time_of_day": time_of_day,
                    "day_of_week": day_of_week,
                    "season": season,
                    "weather": weather,
                    "device_type": device_type,
                    "location_type": location_type,
                }
                interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def _calculate_context_weight(
        self, 
        user: pd.Series, 
        item: pd.Series, 
        time_of_day: str, 
        location_type: str
    ) -> float:
        """Calculate interaction weight based on context match.
        
        Args:
            user: User information.
            item: Item information.
            time_of_day: Current time of day.
            location_type: Current location type.
            
        Returns:
            Weight value between 0 and 1.
        """
        base_weight = np.random.uniform(0.1, 1.0)
        
        # Boost weight if context matches preferences
        if user["preferred_time"] == time_of_day:
            base_weight *= 1.2
        if user["preferred_location"] == location_type:
            base_weight *= 1.2
        if item["time_preference"] == time_of_day:
            base_weight *= 1.1
        if item["location_preference"] == location_type:
            base_weight *= 1.1
            
        return min(base_weight, 1.0)
    
    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset.
        
        Returns:
            Tuple of (interactions_df, items_df, users_df).
        """
        set_random_seeds(self.config.training.random_state)
        
        items_df = self.generate_items()
        users_df = self.generate_users()
        interactions_df = self.generate_interactions(items_df, users_df)
        
        return interactions_df, items_df, users_df


def load_dataset(
    interactions_path: str,
    items_path: str,
    users_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load dataset from files.
    
    Args:
        interactions_path: Path to interactions CSV file.
        items_path: Path to items CSV file.
        users_path: Optional path to users CSV file.
        
    Returns:
        Tuple of loaded DataFrames.
    """
    interactions_df = pd.read_csv(interactions_path)
    items_df = pd.read_csv(items_path)
    users_df = pd.read_csv(users_path) if users_path else None
    
    return interactions_df, items_df, users_df


def save_dataset(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame],
    interactions_path: str,
    items_path: str,
    users_path: Optional[str] = None
) -> None:
    """Save dataset to files.
    
    Args:
        interactions_df: Interactions DataFrame.
        items_df: Items DataFrame.
        users_df: Optional users DataFrame.
        interactions_path: Path to save interactions CSV.
        items_path: Path to save items CSV.
        users_path: Optional path to save users CSV.
    """
    interactions_df.to_csv(interactions_path, index=False)
    items_df.to_csv(items_path, index=False)
    if users_df is not None and users_path is not None:
        users_df.to_csv(users_path, index=False)
