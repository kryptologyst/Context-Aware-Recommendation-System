"""Main training and evaluation script for context-aware recommendation system."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import DatasetGenerator, load_dataset, save_dataset
from src.data.preprocessing import ContextEncoder, DataSplitter, NegativeSampler
from src.evaluation.metrics import ModelEvaluator, create_leaderboard
from src.models.advanced import ALSRecommender, ContentBasedRecommender, LightFMRecommender
from src.models.baselines import ItemKNNRecommender, PopularityRecommender, UserKNNRecommender


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration object.
    """
    return OmegaConf.load(config_path)


def generate_or_load_data(config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Generate synthetic data or load existing data.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (interactions_df, items_df, users_df).
    """
    interactions_path = Path(config.data.processed_data_path) / config.data.interactions_file
    items_path = Path(config.data.processed_data_path) / config.data.items_file
    users_path = Path(config.data.processed_data_path) / config.data.users_file
    
    # Check if data already exists
    if interactions_path.exists() and items_path.exists():
        print("Loading existing data...")
        users_df = None
        if users_path.exists():
            users_df = pd.read_csv(users_path)
        return load_dataset(str(interactions_path), str(items_path), str(users_path) if users_df is not None else None)
    
    # Generate new data
    print("Generating synthetic data...")
    generator = DatasetGenerator(config)
    interactions_df, items_df, users_df = generator.generate_dataset()
    
    # Save generated data
    Path(config.data.processed_data_path).mkdir(parents=True, exist_ok=True)
    save_dataset(
        interactions_df, items_df, users_df,
        str(interactions_path), str(items_path), str(users_path)
    )
    
    print(f"Generated dataset:")
    print(f"  - {len(interactions_df)} interactions")
    print(f"  - {len(items_df)} items")
    print(f"  - {len(users_df)} users")
    
    return interactions_df, items_df, users_df


def create_models(config: DictConfig) -> Dict[str, Any]:
    """Create recommendation models based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Dictionary mapping model names to model instances.
    """
    models = {}
    
    # Baseline models
    if config.models.popularity.enabled:
        models["Popularity"] = PopularityRecommender()
    
    if config.models.user_knn.enabled:
        models["UserKNN"] = UserKNNRecommender(
            k=config.models.user_knn.k,
            min_support=config.models.user_knn.min_support
        )
    
    if config.models.item_knn.enabled:
        models["ItemKNN"] = ItemKNNRecommender(
            k=config.models.item_knn.k,
            min_support=config.models.item_knn.min_support
        )
    
    # Advanced models
    if config.models.als.enabled:
        models["ALS"] = ALSRecommender(
            factors=config.models.als.factors,
            regularization=config.models.als.regularization,
            iterations=config.models.als.iterations
        )
    
    if config.models.content_based.enabled:
        models["ContentBased"] = ContentBasedRecommender(
            use_text_features=config.models.content_based.use_text_features,
            use_categorical_features=config.models.content_based.use_categorical_features,
            tfidf_max_features=config.models.content_based.tfidf_max_features,
            similarity_metric=config.models.content_based.similarity_metric
        )
    
    if config.models.lightfm.enabled:
        models["LightFM"] = LightFMRecommender(
            no_components=config.models.lightfm.no_components,
            learning_rate=config.models.lightfm.learning_rate,
            loss=config.models.lightfm.loss,
            item_alpha=config.models.lightfm.item_alpha,
            user_alpha=config.models.lightfm.user_alpha
        )
    
    return models


def train_models(
    models: Dict[str, Any],
    train_interactions: pd.DataFrame,
    items_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """Train all models.
    
    Args:
        models: Dictionary of models to train.
        train_interactions: Training interactions DataFrame.
        items_df: Items metadata DataFrame.
        users_df: Optional users metadata DataFrame.
        
    Returns:
        Dictionary of trained models.
    """
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        try:
            trained_model = model.fit(train_interactions, items_df, users_df)
            trained_models[model_name] = trained_model
            print(f"✓ {model_name} trained successfully")
        except Exception as e:
            print(f"✗ Failed to train {model_name}: {e}")
    
    return trained_models


def evaluate_models(
    models: Dict[str, Any],
    test_interactions: pd.DataFrame,
    items_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame],
    config: DictConfig
) -> pd.DataFrame:
    """Evaluate all models.
    
    Args:
        models: Dictionary of trained models.
        test_interactions: Test interactions DataFrame.
        items_df: Items metadata DataFrame.
        users_df: Optional users metadata DataFrame.
        config: Configuration object.
        
    Returns:
        DataFrame with evaluation results.
    """
    evaluator = ModelEvaluator(
        metrics=config.evaluation.metrics,
        k_values=config.evaluation.k_values
    )
    
    # Sample context for evaluation
    context = {
        "time_of_day": "afternoon",
        "location_type": "home",
        "device_type": "mobile"
    }
    
    results_df = evaluator.compare_models(
        models, test_interactions, items_df, users_df, context
    )
    
    return results_df


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Context-aware recommendation system")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Configuration file path")
    parser.add_argument("--generate-data", action="store_true", help="Force data generation")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate or load data
    if args.generate_data:
        # Force regeneration
        interactions_path = Path(config.data.processed_data_path) / config.data.interactions_file
        if interactions_path.exists():
            interactions_path.unlink()
    
    interactions_df, items_df, users_df = generate_or_load_data(config)
    
    # Split data
    print("Splitting data...")
    splitter = DataSplitter(
        test_size=config.evaluation.test_size,
        val_size=config.evaluation.val_size,
        random_state=config.evaluation.random_state
    )
    
    train_df, val_df, test_df = splitter.temporal_split(interactions_df)
    
    print(f"Data split:")
    print(f"  - Train: {len(train_df)} interactions")
    print(f"  - Validation: {len(val_df)} interactions")
    print(f"  - Test: {len(test_df)} interactions")
    
    # Create models
    models = create_models(config)
    print(f"Created {len(models)} models: {list(models.keys())}")
    
    if not args.skip_training:
        # Train models
        trained_models = train_models(models, train_df, items_df, users_df)
        
        if not args.skip_evaluation:
            # Evaluate models
            print("Evaluating models...")
            results_df = evaluate_models(trained_models, test_df, items_df, users_df, config)
            
            # Create leaderboard
            leaderboard = create_leaderboard(results_df)
            
            print("\n" + "="*80)
            print("MODEL LEADERBOARD")
            print("="*80)
            print(leaderboard.to_string(index=False))
            
            # Save results
            results_path = Path("results")
            results_path.mkdir(exist_ok=True)
            leaderboard.to_csv(results_path / "leaderboard.csv", index=False)
            print(f"\nResults saved to {results_path / 'leaderboard.csv'}")
    
    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    main()
