"""Streamlit demo for context-aware recommendation system."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import DatasetGenerator, load_dataset
from src.data.preprocessing import DataSplitter
from src.models.advanced import ALSRecommender, ContentBasedRecommender, LightFMRecommender
from src.models.baselines import ItemKNNRecommender, PopularityRecommender, UserKNNRecommender


@st.cache_data
def load_data_and_models() -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """Load data and trained models.
    
    Returns:
        Tuple of (interactions_df, items_df, users_df, trained_models).
    """
    # Load configuration
    config = OmegaConf.load("configs/config.yaml")
    
    # Load data
    interactions_path = Path(config.data.processed_data_path) / config.data.interactions_file
    items_path = Path(config.data.processed_data_path) / config.data.items_file
    users_path = Path(config.data.processed_data_path) / config.data.users_file
    
    if not interactions_path.exists() or not items_path.exists():
        st.error("Data not found. Please run the training script first.")
        st.stop()
    
    users_df = None
    if users_path.exists():
        users_df = pd.read_csv(users_path)
    
    interactions_df, items_df, _ = load_dataset(
        str(interactions_path), str(items_path), str(users_path) if users_df is not None else None
    )
    
    # Create and train models
    models = {
        "Popularity": PopularityRecommender(),
        "UserKNN": UserKNNRecommender(k=50, min_support=5),
        "ItemKNN": ItemKNNRecommender(k=50, min_support=5),
        "ALS": ALSRecommender(factors=50, regularization=0.01, iterations=15),
        "ContentBased": ContentBasedRecommender(),
    }
    
    # Split data for training
    splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
    train_df, _, _ = splitter.temporal_split(interactions_df)
    
    # Train models
    trained_models = {}
    for model_name, model in models.items():
        try:
            trained_model = model.fit(train_df, items_df, users_df)
            trained_models[model_name] = trained_model
        except Exception as e:
            st.warning(f"Failed to train {model_name}: {e}")
    
    return interactions_df, items_df, users_df, trained_models


def get_user_context() -> Dict[str, Any]:
    """Get user context from UI inputs.
    
    Returns:
        Dictionary with context information.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        time_of_day = st.selectbox(
            "Time of Day",
            ["morning", "afternoon", "evening", "night"],
            index=1
        )
        
        location_type = st.selectbox(
            "Location Type",
            ["home", "work", "outdoor", "travel", "restaurant", "gym"],
            index=0
        )
    
    with col2:
        device_type = st.selectbox(
            "Device Type",
            ["mobile", "desktop", "tablet", "tv"],
            index=0
        )
        
        weather = st.selectbox(
            "Weather",
            ["sunny", "cloudy", "rainy", "snowy"],
            index=0
        )
    
    return {
        "time_of_day": time_of_day,
        "location_type": location_type,
        "device_type": device_type,
        "weather": weather
    }


def display_recommendations(
    recommendations: List[str],
    items_df: pd.DataFrame,
    model_name: str,
    n_recommendations: int = 10
) -> None:
    """Display recommendations in a nice format.
    
    Args:
        recommendations: List of recommended item IDs.
        items_df: Items metadata DataFrame.
        model_name: Name of the model.
        n_recommendations: Number of recommendations to display.
    """
    st.subheader(f"Top {n_recommendations} Recommendations ({model_name})")
    
    if not recommendations:
        st.warning("No recommendations available.")
        return
    
    # Create columns for recommendations
    cols = st.columns(min(2, len(recommendations)))
    
    for i, item_id in enumerate(recommendations[:n_recommendations]):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            # Get item information
            item_info = items_df[items_df["item_id"] == item_id]
            if not item_info.empty:
                item = item_info.iloc[0]
                
                # Create recommendation card
                with st.container():
                    st.markdown(f"**{item['title']}**")
                    st.markdown(f"*Category: {item['category']}*")
                    st.markdown(f"*Price: ${item['price']:.2f}*")
                    st.markdown(f"*Rating: {item['rating']:.1f}/5.0*")
                    
                    # Show context preferences
                    if "time_preference" in item:
                        st.markdown(f"*Best for: {item['time_preference']}*")
                    if "location_preference" in item:
                        st.markdown(f"*Location: {item['location_preference']}*")
                    
                    st.markdown("---")


def display_similar_items(
    similar_items: List[tuple[str, float]],
    items_df: pd.DataFrame,
    n_similar: int = 5
) -> None:
    """Display similar items.
    
    Args:
        similar_items: List of (item_id, similarity_score) tuples.
        items_df: Items metadata DataFrame.
        n_similar: Number of similar items to display.
    """
    st.subheader(f"Similar Items")
    
    if not similar_items:
        st.warning("No similar items found.")
        return
    
    for item_id, similarity in similar_items[:n_similar]:
        item_info = items_df[items_df["item_id"] == item_id]
        if not item_info.empty:
            item = item_info.iloc[0]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{item['title']}**")
                st.markdown(f"*Category: {item['category']}*")
            with col2:
                st.metric("Similarity", f"{similarity:.3f}")


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Context-Aware Recommendation System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Context-Aware Recommendation System")
    st.markdown("A modern recommendation system that considers contextual information like time, location, and device type.")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        interactions_df, items_df, users_df, trained_models = load_data_and_models()
    
    if not trained_models:
        st.error("No models available. Please check the training process.")
        st.stop()
    
    # Sidebar for user selection and context
    st.sidebar.header("User & Context Selection")
    
    # User selection
    available_users = interactions_df["user_id"].unique().tolist()
    selected_user = st.sidebar.selectbox(
        "Select User",
        available_users,
        index=0
    )
    
    # Context selection
    st.sidebar.subheader("Context Information")
    context = get_user_context()
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        list(trained_models.keys()),
        default=list(trained_models.keys())[:2]
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )
    
    # Main content
    st.header("Recommendations")
    
    if selected_models:
        # Generate recommendations for each selected model
        for model_name in selected_models:
            if model_name in trained_models:
                model = trained_models[model_name]
                
                try:
                    recommendations = model.predict(
                        selected_user,
                        n_recommendations=n_recommendations,
                        context=context
                    )
                    
                    display_recommendations(recommendations, items_df, model_name, n_recommendations)
                    
                except Exception as e:
                    st.error(f"Error generating recommendations with {model_name}: {e}")
    
    # Item similarity section
    st.header("Item Similarity")
    
    # Item selection for similarity
    available_items = items_df["item_id"].tolist()
    selected_item = st.selectbox(
        "Select Item to Find Similar Items",
        available_items,
        index=0
    )
    
    # Model for similarity
    similarity_model = st.selectbox(
        "Select Model for Similarity",
        list(trained_models.keys()),
        index=0
    )
    
    if st.button("Find Similar Items"):
        model = trained_models[similarity_model]
        try:
            similar_items = model.get_similar_items(selected_item, n_similar=10)
            display_similar_items(similar_items, items_df)
        except Exception as e:
            st.error(f"Error finding similar items: {e}")
    
    # Dataset statistics
    st.header("Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", len(interactions_df["user_id"].unique()))
    
    with col2:
        st.metric("Total Items", len(items_df))
    
    with col3:
        st.metric("Total Interactions", len(interactions_df))
    
    # Context distribution
    st.subheader("Context Distribution")
    
    context_cols = ["time_of_day", "location_type", "device_type"]
    for col in context_cols:
        if col in interactions_df.columns:
            st.bar_chart(interactions_df[col].value_counts())


if __name__ == "__main__":
    main()
