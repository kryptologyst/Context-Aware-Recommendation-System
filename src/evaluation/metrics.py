"""Evaluation metrics and model comparison utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


class RecommendationMetrics:
    """Calculate various recommendation metrics."""
    
    def __init__(self) -> None:
        """Initialize metrics calculator."""
        pass
    
    def precision_at_k(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        k: int
    ) -> float:
        """Calculate Precision@K.
        
        Args:
            y_true: True item IDs.
            y_pred: Predicted item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Precision@K score.
        """
        if len(y_pred) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        hits = len(set(y_true) & set(y_pred_k))
        return hits / min(k, len(y_pred_k))
    
    def recall_at_k(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        k: int
    ) -> float:
        """Calculate Recall@K.
        
        Args:
            y_true: True item IDs.
            y_pred: Predicted item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Recall@K score.
        """
        if len(y_true) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        hits = len(set(y_true) & set(y_pred_k))
        return hits / len(y_true)
    
    def hit_rate_at_k(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        k: int
    ) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            y_true: True item IDs.
            y_pred: Predicted item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Hit Rate@K score.
        """
        y_pred_k = y_pred[:k]
        hits = len(set(y_true) & set(y_pred_k))
        return 1.0 if hits > 0 else 0.0
    
    def ndcg_at_k(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        k: int
    ) -> float:
        """Calculate NDCG@K.
        
        Args:
            y_true: True item IDs.
            y_pred: Predicted item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            NDCG@K score.
        """
        if len(y_pred) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(y_pred_k):
            if item in y_true:
                dcg += 1.0 / np.log2(i + 2)
        
        # Calculate IDCG
        idcg = 0.0
        for i in range(min(len(y_true), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def map_at_k(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        k: int
    ) -> float:
        """Calculate MAP@K.
        
        Args:
            y_true: True item IDs.
            y_pred: Predicted item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            MAP@K score.
        """
        if len(y_pred) == 0 or len(y_true) == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        
        # Calculate average precision
        precision_sum = 0.0
        hits = 0
        
        for i, item in enumerate(y_pred_k):
            if item in y_true:
                hits += 1
                precision_sum += hits / (i + 1)
        
        return precision_sum / len(y_true)
    
    def coverage(
        self, 
        recommendations: Dict[str, List[str]], 
        all_items: List[str]
    ) -> float:
        """Calculate catalog coverage.
        
        Args:
            recommendations: Dictionary mapping user IDs to recommendations.
            all_items: List of all available items.
            
        Returns:
            Coverage score.
        """
        if not recommendations:
            return 0.0
        
        recommended_items = set()
        for user_recs in recommendations.values():
            recommended_items.update(user_recs)
        
        return len(recommended_items) / len(all_items)
    
    def diversity(
        self, 
        recommendations: Dict[str, List[str]], 
        item_features: Optional[pd.DataFrame] = None
    ) -> float:
        """Calculate recommendation diversity.
        
        Args:
            recommendations: Dictionary mapping user IDs to recommendations.
            item_features: Optional item features DataFrame.
            
        Returns:
            Diversity score.
        """
        if not recommendations:
            return 0.0
        
        # Simple diversity based on unique items
        all_recommended_items = []
        for user_recs in recommendations.values():
            all_recommended_items.extend(user_recs)
        
        unique_items = len(set(all_recommended_items))
        total_items = len(all_recommended_items)
        
        return unique_items / total_items if total_items > 0 else 0.0
    
    def novelty(
        self, 
        recommendations: Dict[str, List[str]], 
        item_popularity: Dict[str, float]
    ) -> float:
        """Calculate recommendation novelty.
        
        Args:
            recommendations: Dictionary mapping user IDs to recommendations.
            item_popularity: Dictionary mapping item IDs to popularity scores.
            
        Returns:
            Novelty score.
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for user_recs in recommendations.values():
            for item in user_recs:
                # Novelty is inverse of popularity
                popularity = item_popularity.get(item, 0.0)
                novelty = 1.0 - popularity
                novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0


class ModelEvaluator:
    """Evaluate recommendation models."""
    
    def __init__(self, metrics: Optional[List[str]] = None, k_values: Optional[List[int]] = None) -> None:
        """Initialize model evaluator.
        
        Args:
            metrics: List of metrics to calculate.
            k_values: List of K values for ranking metrics.
        """
        self.metrics = metrics or ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k", "map_at_k"]
        self.k_values = k_values or [5, 10, 20]
        self.metrics_calculator = RecommendationMetrics()
    
    def evaluate_model(
        self,
        model: Any,
        test_interactions: pd.DataFrame,
        items_df: Optional[pd.DataFrame] = None,
        users_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate a single model.
        
        Args:
            model: Recommendation model to evaluate.
            test_interactions: Test interactions DataFrame.
            items_df: Optional items metadata DataFrame.
            users_df: Optional users metadata DataFrame.
            context: Optional context information.
            
        Returns:
            Dictionary of metric scores.
        """
        results = {}
        
        # Group test interactions by user
        user_test_items = test_interactions.groupby("user_id")["item_id"].apply(list).to_dict()
        
        # Generate recommendations for all users
        all_users = list(user_test_items.keys())
        recommendations = model.predict(all_users, n_recommendations=max(self.k_values), context=context)
        
        # Calculate metrics for each K value
        for metric in self.metrics:
            if metric.endswith("_at_k"):
                for k in self.k_values:
                    metric_name = f"{metric}_{k}"
                    scores = []
                    
                    for user_id in all_users:
                        if user_id in recommendations and user_id in user_test_items:
                            y_true = user_test_items[user_id]
                            y_pred = recommendations[user_id]
                            
                            metric_func = getattr(self.metrics_calculator, metric)
                            score = metric_func(y_true, y_pred, k)
                            scores.append(score)
                    
                    results[metric_name] = np.mean(scores) if scores else 0.0
        
        # Calculate additional metrics
        if "coverage" in self.metrics and items_df is not None:
            all_items = items_df["item_id"].tolist()
            results["coverage"] = self.metrics_calculator.coverage(recommendations, all_items)
        
        if "diversity" in self.metrics:
            results["diversity"] = self.metrics_calculator.diversity(recommendations)
        
        if "novelty" in self.metrics and items_df is not None:
            # Calculate item popularity from test data
            item_popularity = test_interactions.groupby("item_id").size()
            item_popularity = item_popularity / item_popularity.max()
            item_popularity_dict = item_popularity.to_dict()
            
            results["novelty"] = self.metrics_calculator.novelty(recommendations, item_popularity_dict)
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_interactions: pd.DataFrame,
        items_df: Optional[pd.DataFrame] = None,
        users_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances.
            test_interactions: Test interactions DataFrame.
            items_df: Optional items metadata DataFrame.
            users_df: Optional users metadata DataFrame.
            context: Optional context information.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            model_results = self.evaluate_model(
                model, test_interactions, items_df, users_df, context
            )
            model_results["model"] = model_name
            results.append(model_results)
        
        return pd.DataFrame(results)


def create_leaderboard(
    results_df: pd.DataFrame,
    primary_metric: str = "ndcg_at_k_10",
    secondary_metric: str = "precision_at_k_10"
) -> pd.DataFrame:
    """Create a model leaderboard.
    
    Args:
        results_df: Results DataFrame from model comparison.
        primary_metric: Primary metric for ranking.
        secondary_metric: Secondary metric for tie-breaking.
        
    Returns:
        Sorted leaderboard DataFrame.
    """
    if primary_metric not in results_df.columns:
        raise ValueError(f"Primary metric '{primary_metric}' not found in results")
    
    # Sort by primary metric, then by secondary metric
    sort_columns = [primary_metric]
    if secondary_metric in results_df.columns:
        sort_columns.append(secondary_metric)
    
    leaderboard = results_df.sort_values(sort_columns, ascending=False)
    
    # Add rank
    leaderboard["rank"] = range(1, len(leaderboard) + 1)
    
    return leaderboard
