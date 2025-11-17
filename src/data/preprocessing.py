"""Data preprocessing and splitting utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Handle data splitting for recommendation systems."""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> None:
        """Initialize data splitter.
        
        Args:
            test_size: Proportion of data for testing.
            val_size: Proportion of data for validation.
            random_state: Random seed for reproducibility.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def temporal_split(
        self, 
        interactions_df: pd.DataFrame,
        user_col: str = "user_id",
        time_col: str = "timestamp"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally (chronological split).
        
        Args:
            interactions_df: Interactions DataFrame.
            user_col: User column name.
            time_col: Timestamp column name.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # Sort by user and timestamp
        df_sorted = interactions_df.sort_values([user_col, time_col])
        
        train_data = []
        val_data = []
        test_data = []
        
        for user_id in df_sorted[user_col].unique():
            user_interactions = df_sorted[df_sorted[user_col] == user_id]
            
            if len(user_interactions) < 3:
                # If user has less than 3 interactions, put all in train
                train_data.append(user_interactions)
                continue
            
            # Split chronologically
            n_interactions = len(user_interactions)
            train_size = int(n_interactions * (1 - self.test_size - self.val_size))
            val_size = int(n_interactions * self.val_size)
            
            train_data.append(user_interactions.iloc[:train_size])
            val_data.append(user_interactions.iloc[train_size:train_size + val_size])
            test_data.append(user_interactions.iloc[train_size + val_size:])
        
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        return train_df, val_df, test_df
    
    def random_split(
        self, 
        interactions_df: pd.DataFrame,
        user_col: str = "user_id"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data randomly.
        
        Args:
            interactions_df: Interactions DataFrame.
            user_col: User column name.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # First split into train+val and test
        train_val_df, test_df = train_test_split(
            interactions_df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=interactions_df[user_col]
        )
        
        # Then split train+val into train and val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
            stratify=train_val_df[user_col]
        )
        
        return train_df, val_df, test_df


class NegativeSampler:
    """Generate negative samples for implicit feedback."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize negative sampler.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def sample_negatives(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame,
        n_negatives: int = 1,
        user_col: str = "user_id",
        item_col: str = "item_id"
    ) -> pd.DataFrame:
        """Sample negative interactions.
        
        Args:
            interactions_df: Positive interactions DataFrame.
            items_df: Items DataFrame.
            n_negatives: Number of negative samples per positive.
            user_col: User column name.
            item_col: Item column name.
            
        Returns:
            DataFrame with negative interactions.
        """
        negative_interactions = []
        
        # Get all items
        all_items = set(items_df[item_col].unique())
        
        for _, interaction in interactions_df.iterrows():
            user_id = interaction[user_col]
            
            # Get items this user has interacted with
            user_items = set(
                interactions_df[interactions_df[user_col] == user_id][item_col].unique()
            )
            
            # Sample negative items
            negative_items = all_items - user_items
            if len(negative_items) == 0:
                continue
                
            n_samples = min(n_negatives, len(negative_items))
            sampled_negatives = np.random.choice(
                list(negative_items), 
                size=n_samples, 
                replace=False
            )
            
            for neg_item in sampled_negatives:
                neg_interaction = interaction.copy()
                neg_interaction[item_col] = neg_item
                neg_interaction["weight"] = 0.0  # Negative interaction
                neg_interaction["is_positive"] = False
                negative_interactions.append(neg_interaction)
        
        return pd.DataFrame(negative_interactions)


class ContextEncoder:
    """Encode contextual features for models."""
    
    def __init__(self) -> None:
        """Initialize context encoder."""
        self.context_features = [
            "time_of_day", "day_of_week", "season", 
            "weather", "device_type", "location_type"
        ]
        self.encoders = {}
    
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit encoders on training data.
        
        Args:
            interactions_df: Training interactions DataFrame.
        """
        from sklearn.preprocessing import LabelEncoder
        
        for feature in self.context_features:
            if feature in interactions_df.columns:
                encoder = LabelEncoder()
                encoder.fit(interactions_df[feature])
                self.encoders[feature] = encoder
    
    def transform(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Transform context features to numerical.
        
        Args:
            interactions_df: Interactions DataFrame.
            
        Returns:
            DataFrame with encoded context features.
        """
        df_encoded = interactions_df.copy()
        
        for feature in self.context_features:
            if feature in interactions_df.columns and feature in self.encoders:
                df_encoded[f"{feature}_encoded"] = self.encoders[feature].transform(
                    interactions_df[feature]
                )
        
        return df_encoded
    
    def fit_transform(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders and transform data.
        
        Args:
            interactions_df: Interactions DataFrame.
            
        Returns:
            DataFrame with encoded context features.
        """
        self.fit(interactions_df)
        return self.transform(interactions_df)
