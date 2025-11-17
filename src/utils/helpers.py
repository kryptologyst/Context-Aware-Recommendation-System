"""Utility functions for the recommendation system."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Save a trained model to disk.
    
    Args:
        model: Trained model object.
        filepath: Path to save the model.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: Union[str, Path]) -> Any:
    """Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model.
        
    Returns:
        Loaded model object.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_results(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary.
        filepath: Path to save results.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy(obj)
    
    converted_results = recursive_convert(results)
    
    with open(filepath, "w") as f:
        json.dump(converted_results, f, indent=2)


def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation results from JSON file.
    
    Args:
        filepath: Path to results file.
        
    Returns:
        Results dictionary.
    """
    with open(filepath, "r") as f:
        return json.load(f)


def create_experiment_summary(
    config: Dict[str, Any],
    results: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a comprehensive experiment summary.
    
    Args:
        config: Configuration used for the experiment.
        results: Evaluation results.
        model_info: Optional model information.
        
    Returns:
        Experiment summary dictionary.
    """
    summary = {
        "experiment_id": f"exp_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": pd.Timestamp.now().isoformat(),
        "config": config,
        "results": results,
        "model_info": model_info or {},
        "summary": {
            "best_model": None,
            "best_score": 0.0,
            "total_models": len(results.get("model_results", [])),
        }
    }
    
    # Find best model based on primary metric
    if "model_results" in results:
        primary_metric = "ndcg_at_k_10"  # Default primary metric
        best_score = 0.0
        best_model = None
        
        for model_result in results["model_results"]:
            if primary_metric in model_result:
                score = model_result[primary_metric]
                if score > best_score:
                    best_score = score
                    best_model = model_result.get("model", "unknown")
        
        summary["summary"]["best_model"] = best_model
        summary["summary"]["best_score"] = best_score
    
    return summary


def print_experiment_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted experiment summary.
    
    Args:
        summary: Experiment summary dictionary.
    """
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Experiment ID: {summary['experiment_id']}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Total Models: {summary['summary']['total_models']}")
    print(f"Best Model: {summary['summary']['best_model']}")
    print(f"Best Score: {summary['summary']['best_score']:.4f}")
    print("=" * 80)


def validate_data_quality(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Validate data quality and return quality metrics.
    
    Args:
        interactions_df: Interactions DataFrame.
        items_df: Items DataFrame.
        users_df: Optional users DataFrame.
        
    Returns:
        Dictionary with quality metrics.
    """
    quality_metrics = {
        "interactions": {
            "total_count": len(interactions_df),
            "unique_users": interactions_df["user_id"].nunique(),
            "unique_items": interactions_df["item_id"].nunique(),
            "sparsity": 1.0 - (len(interactions_df) / (interactions_df["user_id"].nunique() * interactions_df["item_id"].nunique())),
            "avg_interactions_per_user": len(interactions_df) / interactions_df["user_id"].nunique(),
            "avg_interactions_per_item": len(interactions_df) / interactions_df["item_id"].nunique(),
        },
        "items": {
            "total_count": len(items_df),
            "categories": items_df["category"].nunique() if "category" in items_df.columns else 0,
            "missing_values": items_df.isnull().sum().to_dict(),
        },
        "users": {
            "total_count": len(users_df) if users_df is not None else 0,
            "missing_values": users_df.isnull().sum().to_dict() if users_df is not None else {},
        }
    }
    
    # Check for data issues
    issues = []
    
    # Check for duplicate interactions
    duplicates = interactions_df.duplicated(subset=["user_id", "item_id"]).sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate user-item interactions")
    
    # Check for users with no interactions
    users_with_interactions = set(interactions_df["user_id"].unique())
    if users_df is not None:
        all_users = set(users_df["user_id"].unique())
        users_without_interactions = all_users - users_with_interactions
        if users_without_interactions:
            issues.append(f"Found {len(users_without_interactions)} users with no interactions")
    
    # Check for items with no interactions
    items_with_interactions = set(interactions_df["item_id"].unique())
    all_items = set(items_df["item_id"].unique())
    items_without_interactions = all_items - items_with_interactions
    if items_without_interactions:
        issues.append(f"Found {len(items_without_interactions)} items with no interactions")
    
    quality_metrics["issues"] = issues
    quality_metrics["quality_score"] = max(0, 1.0 - len(issues) * 0.1)
    
    return quality_metrics


def print_data_quality_report(quality_metrics: Dict[str, Any]) -> None:
    """Print a formatted data quality report.
    
    Args:
        quality_metrics: Quality metrics dictionary.
    """
    print("=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    
    # Interactions
    interactions = quality_metrics["interactions"]
    print(f"Interactions: {interactions['total_count']:,}")
    print(f"Unique Users: {interactions['unique_users']:,}")
    print(f"Unique Items: {interactions['unique_items']:,}")
    print(f"Sparsity: {interactions['sparsity']:.4f}")
    print(f"Avg Interactions per User: {interactions['avg_interactions_per_user']:.2f}")
    print(f"Avg Interactions per Item: {interactions['avg_interactions_per_item']:.2f}")
    
    # Items
    items = quality_metrics["items"]
    print(f"\nItems: {items['total_count']:,}")
    print(f"Categories: {items['categories']}")
    
    # Users
    users = quality_metrics["users"]
    if users["total_count"] > 0:
        print(f"\nUsers: {users['total_count']:,}")
    
    # Issues
    issues = quality_metrics["issues"]
    if issues:
        print(f"\nIssues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nNo issues found!")
    
    print(f"\nQuality Score: {quality_metrics['quality_score']:.2f}")
    print("=" * 80)
