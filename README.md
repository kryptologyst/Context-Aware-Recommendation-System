# Context-Aware Recommendation System

A production-ready context-aware recommendation system that considers contextual information like time of day, location, device type, and weather to provide personalized recommendations.

## Features

- **Context-Aware Recommendations**: Considers time, location, device, and weather context
- **Multiple Model Types**: Popularity, collaborative filtering, matrix factorization, content-based, and hybrid models
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Coverage, Diversity, and Novelty metrics
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Production Ready**: Clean code with type hints, comprehensive testing, and proper documentation
- **Configurable**: YAML-based configuration for easy experimentation

## Project Structure

```
├── src/                          # Source code
│   ├── data/                     # Data handling modules
│   │   ├── dataset.py           # Dataset generation and loading
│   │   └── preprocessing.py     # Data preprocessing and splitting
│   ├── models/                   # Recommendation models
│   │   ├── base.py             # Base classes and interfaces
│   │   ├── baselines.py        # Baseline models (Popularity, KNN)
│   │   └── advanced.py         # Advanced models (ALS, Content-based, LightFM)
│   ├── evaluation/              # Evaluation metrics and utilities
│   │   └── metrics.py          # Recommendation metrics and model comparison
│   └── utils/                   # Utility functions
├── configs/                      # Configuration files
│   └── config.yaml             # Main configuration
├── data/                        # Data directory
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── scripts/                     # Executable scripts
│   ├── train_evaluate.py       # Training and evaluation script
│   └── demo.py                # Streamlit demo
├── tests/                       # Test files
├── notebooks/                   # Jupyter notebooks for exploration
├── assets/                      # Static assets
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Context-Aware-Recommendation-System.git
cd Context-Aware-Recommendation-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data and Train Models

```bash
# Run training and evaluation
python scripts/train_evaluate.py

# Or with custom configuration
python scripts/train_evaluate.py --config configs/config.yaml
```

### 3. Launch Interactive Demo

```bash
# Start Streamlit demo
streamlit run scripts/demo.py

# The demo will be available at http://localhost:8501
```

## Configuration

The system is configured via `configs/config.yaml`. Key configuration options:

### Data Configuration
- `n_users`: Number of users in synthetic dataset
- `n_items`: Number of items in synthetic dataset
- `n_interactions`: Total number of interactions
- `context_features`: List of context features to use

### Model Configuration
- Enable/disable specific models
- Set hyperparameters for each model
- Configure evaluation metrics and K values

### Evaluation Configuration
- Test/validation split ratios
- Metrics to calculate
- K values for ranking metrics

## Models

### Baseline Models
- **Popularity**: Recommends most popular items
- **UserKNN**: User-based collaborative filtering
- **ItemKNN**: Item-based collaborative filtering

### Advanced Models
- **ALS**: Alternating Least Squares matrix factorization
- **Content-Based**: TF-IDF based content recommendations
- **LightFM**: Hybrid matrix factorization with features

## Context Features

The system considers the following contextual information:

- **Time of Day**: morning, afternoon, evening, night
- **Day of Week**: monday through sunday
- **Season**: spring, summer, autumn, winter
- **Weather**: sunny, cloudy, rainy, snowy
- **Device Type**: mobile, desktop, tablet, tv
- **Location Type**: home, work, outdoor, travel, restaurant, gym

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog that is recommended
- **Diversity**: Measure of recommendation diversity
- **Novelty**: Measure of recommendation novelty

## Usage Examples

### Basic Training and Evaluation

```python
from omegaconf import OmegaConf
from src.data.dataset import DatasetGenerator
from src.models.baselines import PopularityRecommender
from src.evaluation.metrics import ModelEvaluator

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Generate data
generator = DatasetGenerator(config)
interactions_df, items_df, users_df = generator.generate_dataset()

# Create and train model
model = PopularityRecommender()
model.fit(interactions_df, items_df, users_df)

# Generate recommendations
recommendations = model.predict("user_0001", n_recommendations=10)
print(recommendations)
```

### Context-Aware Recommendations

```python
# Define context
context = {
    "time_of_day": "evening",
    "location_type": "home",
    "device_type": "mobile",
    "weather": "rainy"
}

# Generate context-aware recommendations
recommendations = model.predict(
    "user_0001", 
    n_recommendations=10, 
    context=context
)
```

### Model Comparison

```python
from src.evaluation.metrics import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator(
    metrics=["precision_at_k", "recall_at_k", "ndcg_at_k"],
    k_values=[5, 10, 20]
)

# Compare models
results_df = evaluator.compare_models(
    models={"Popularity": model1, "UserKNN": model2},
    test_interactions=test_df,
    items_df=items_df,
    users_df=users_df
)

print(results_df)
```

## Interactive Demo

The Streamlit demo provides:

1. **User Selection**: Choose from available users
2. **Context Configuration**: Set time, location, device, and weather
3. **Model Comparison**: Compare recommendations across different models
4. **Item Similarity**: Find similar items for any given item
5. **Dataset Statistics**: View dataset overview and context distributions

### Demo Features

- Real-time recommendation generation
- Context-aware filtering
- Model performance comparison
- Interactive item similarity exploration
- Dataset visualization

## Development

### Code Quality

The project uses modern Python development practices:

- **Type Hints**: Full type annotation support
- **Code Formatting**: Black and Ruff for consistent formatting
- **Testing**: Pytest for unit tests
- **Documentation**: Google-style docstrings

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

### Code Formatting

```bash
# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ scripts/ tests/
```

## Extending the System

### Adding New Models

1. Inherit from `BaseRecommender` or `ContextAwareRecommender`
2. Implement required methods: `fit`, `predict`, `get_similar_items`
3. Add model configuration to `config.yaml`
4. Register model in `create_models()` function

### Adding New Context Features

1. Add feature to `ContextFeatures` class
2. Update data generation in `DatasetGenerator`
3. Modify context encoding in `ContextEncoder`
4. Update demo UI for new feature

### Adding New Metrics

1. Implement metric function in `RecommendationMetrics`
2. Add metric to `ModelEvaluator` configuration
3. Update evaluation pipeline

## Performance Considerations

- **Memory Usage**: Large datasets may require chunked processing
- **Model Training**: Some models (LightFM) may take longer to train
- **Recommendation Speed**: Consider caching for production use
- **Scalability**: For production, consider distributed training

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce dataset size or use chunked processing
3. **Model Training Failures**: Check data quality and model parameters
4. **Demo Not Loading**: Ensure data files exist in `data/processed/`

### Getting Help

- Check the configuration file for parameter descriptions
- Review model docstrings for usage examples
- Run tests to verify installation
- Check logs for detailed error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Citation

If you use this system in your research, please cite:

```bibtex
@software{context_aware_recommendation_system,
  title={Context-Aware Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Context-Aware-Recommendation-System}
}
```
# Context-Aware-Recommendation-System
