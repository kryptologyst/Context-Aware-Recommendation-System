"""Test cases for the recommendation system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.data.dataset import DatasetGenerator, ContextFeatures
from src.data.preprocessing import DataSplitter, NegativeSampler, ContextEncoder
from src.models.baselines import PopularityRecommender, UserKNNRecommender, ItemKNNRecommender
from src.models.advanced import ALSRecommender, ContentBasedRecommender
from src.evaluation.metrics import RecommendationMetrics, ModelEvaluator
from src.utils.helpers import validate_data_quality


class TestDatasetGenerator:
    """Test cases for DatasetGenerator."""
    
    def test_context_features(self):
        """Test ContextFeatures class."""
        assert len(ContextFeatures.TIME_OF_DAY) == 4
        assert len(ContextFeatures.DAY_OF_WEEK) == 7
        assert len(ContextFeatures.SEASONS) == 4
        assert "morning" in ContextFeatures.TIME_OF_DAY
        assert "monday" in ContextFeatures.DAY_OF_WEEK
    
    def test_generate_items(self):
        """Test item generation."""
        config = Mock()
        config.data.n_items = 10
        
        generator = DatasetGenerator(config)
        items_df = generator.generate_items()
        
        assert len(items_df) == 10
        assert "item_id" in items_df.columns
        assert "title" in items_df.columns
        assert "category" in items_df.columns
        assert all(items_df["item_id"].str.startswith("item_"))
    
    def test_generate_users(self):
        """Test user generation."""
        config = Mock()
        config.data.n_users = 5
        
        generator = DatasetGenerator(config)
        users_df = generator.generate_users()
        
        assert len(users_df) == 5
        assert "user_id" in users_df.columns
        assert "age" in users_df.columns
        assert "gender" in users_df.columns
        assert all(users_df["user_id"].str.startswith("user_"))


class TestDataSplitter:
    """Test cases for DataSplitter."""
    
    def test_temporal_split(self):
        """Test temporal data splitting."""
        # Create test data
        data = []
        for user_id in ["user1", "user2"]:
            for i in range(10):
                data.append({
                    "user_id": user_id,
                    "item_id": f"item_{i}",
                    "timestamp": pd.Timestamp(f"2024-01-{i+1:02d}"),
                    "weight": 1.0
                })
        
        df = pd.DataFrame(data)
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        
        train_df, val_df, test_df = splitter.temporal_split(df)
        
        # Check that splits are chronological
        assert len(train_df) > len(val_df)
        assert len(val_df) > len(test_df)
        assert len(train_df) + len(val_df) + len(test_df) == len(df)


class TestRecommendationMetrics:
    """Test cases for RecommendationMetrics."""
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        metrics = RecommendationMetrics()
        
        y_true = ["item1", "item2", "item3"]
        y_pred = ["item1", "item4", "item2", "item5"]
        
        precision = metrics.precision_at_k(y_true, y_pred, k=3)
        assert precision == 2/3  # 2 hits out of 3 predictions
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        metrics = RecommendationMetrics()
        
        y_true = ["item1", "item2", "item3"]
        y_pred = ["item1", "item4", "item2", "item5"]
        
        recall = metrics.recall_at_k(y_true, y_pred, k=3)
        assert recall == 2/3  # 2 hits out of 3 true items
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        metrics = RecommendationMetrics()
        
        y_true = ["item1", "item2"]
        y_pred = ["item1", "item3", "item2", "item4"]
        
        ndcg = metrics.ndcg_at_k(y_true, y_pred, k=3)
        assert 0 <= ndcg <= 1
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        metrics = RecommendationMetrics()
        
        y_true = ["item1", "item2"]
        y_pred = ["item1", "item3", "item4"]
        
        hit_rate = metrics.hit_rate_at_k(y_true, y_pred, k=3)
        assert hit_rate == 1.0  # At least one hit


class TestPopularityRecommender:
    """Test cases for PopularityRecommender."""
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        # Create test data
        interactions_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user2", "user2"],
            "item_id": ["item1", "item2", "item1", "item3"],
            "weight": [1.0, 1.0, 1.0, 1.0]
        })
        
        items_df = pd.DataFrame({
            "item_id": ["item1", "item2", "item3"],
            "title": ["Item 1", "Item 2", "Item 3"],
            "category": ["A", "B", "A"]
        })
        
        model = PopularityRecommender()
        model.fit(interactions_df, items_df)
        
        # Test prediction
        recommendations = model.predict("user1", n_recommendations=2)
        assert len(recommendations) <= 2
        assert all(item in ["item1", "item2", "item3"] for item in recommendations)
    
    def test_get_similar_items(self):
        """Test similar items retrieval."""
        interactions_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user2"],
            "item_id": ["item1", "item2", "item1"],
            "weight": [1.0, 1.0, 1.0]
        })
        
        items_df = pd.DataFrame({
            "item_id": ["item1", "item2", "item3"],
            "title": ["Item 1", "Item 2", "Item 3"],
            "category": ["A", "B", "A"]
        })
        
        model = PopularityRecommender()
        model.fit(interactions_df, items_df)
        
        similar_items = model.get_similar_items("item1", n_similar=2)
        assert len(similar_items) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similar_items)


class TestUserKNNRecommender:
    """Test cases for UserKNNRecommender."""
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        # Create test data with more interactions for meaningful similarity
        interactions_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user1", "user2", "user2", "user2"],
            "item_id": ["item1", "item2", "item3", "item1", "item2", "item4"],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        items_df = pd.DataFrame({
            "item_id": ["item1", "item2", "item3", "item4"],
            "title": ["Item 1", "Item 2", "Item 3", "Item 4"],
            "category": ["A", "B", "A", "B"]
        })
        
        model = UserKNNRecommender(k=2, min_support=1)
        model.fit(interactions_df, items_df)
        
        # Test prediction
        recommendations = model.predict("user1", n_recommendations=2)
        assert len(recommendations) <= 2


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender."""
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        interactions_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user2"],
            "item_id": ["item1", "item2", "item1"],
            "weight": [1.0, 1.0, 1.0]
        })
        
        items_df = pd.DataFrame({
            "item_id": ["item1", "item2", "item3"],
            "title": ["Action Movie", "Comedy Show", "Action Series"],
            "category": ["movie", "show", "series"]
        })
        
        model = ContentBasedRecommender()
        model.fit(interactions_df, items_df)
        
        # Test prediction
        recommendations = model.predict("user1", n_recommendations=2)
        assert len(recommendations) <= 2


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create test data
        train_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user2"],
            "item_id": ["item1", "item2", "item1"],
            "weight": [1.0, 1.0, 1.0]
        })
        
        test_df = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "item_id": ["item3", "item2"],
            "weight": [1.0, 1.0]
        })
        
        items_df = pd.DataFrame({
            "item_id": ["item1", "item2", "item3"],
            "title": ["Item 1", "Item 2", "Item 3"],
            "category": ["A", "B", "A"]
        })
        
        # Train model
        model = PopularityRecommender()
        model.fit(train_df, items_df)
        
        # Evaluate model
        evaluator = ModelEvaluator(metrics=["precision_at_k"], k_values=[5])
        results = evaluator.evaluate_model(model, test_df, items_df)
        
        assert "precision_at_k_5" in results
        assert 0 <= results["precision_at_k_5"] <= 1


class TestDataQualityValidation:
    """Test cases for data quality validation."""
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        interactions_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user2"],
            "item_id": ["item1", "item2", "item1"],
            "weight": [1.0, 1.0, 1.0]
        })
        
        items_df = pd.DataFrame({
            "item_id": ["item1", "item2"],
            "title": ["Item 1", "Item 2"],
            "category": ["A", "B"]
        })
        
        users_df = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "age": [25, 30],
            "gender": ["M", "F"]
        })
        
        quality_metrics = validate_data_quality(interactions_df, items_df, users_df)
        
        assert "interactions" in quality_metrics
        assert "items" in quality_metrics
        assert "users" in quality_metrics
        assert "issues" in quality_metrics
        assert "quality_score" in quality_metrics
        
        assert quality_metrics["interactions"]["total_count"] == 3
        assert quality_metrics["interactions"]["unique_users"] == 2
        assert quality_metrics["interactions"]["unique_items"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
