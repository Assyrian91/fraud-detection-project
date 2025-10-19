"""
Script to train fraud detection models
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import Config
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.train import ModelTrainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train fraud detection model')

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to input data file'
    )

    parser.add_argument(
        '--balance-method',
        type=str,
        default='smote',
        choices=['smote', 'undersample', 'none'],
        help='Method to handle class imbalance'
    )

    parser.add_argument(
        '--feature-engineering',
        action='store_true',
        help='Apply feature engineering'
    )

    parser.add_argument(
        '--hyperparameter-tuning',
        action='store_true',
        help='Perform hyperparameter tuning'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        choices=['all', 'logistic', 'rf', 'xgboost', 'lightgbm', 'gb'],
        help='Models to train'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save trained models'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    logger.info("="*50)
    logger.info("Starting Fraud Detection Model Training")
    logger.info("="*50)

    # Initialize components
    config = Config()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()

    # Load data
    logger.info("Step 1: Loading data...")
    data_path = args.data_path if args.data_path else config.RAW_DATA_FILE
    df = preprocessor.load_data(data_path)
    logger.info(f"Data loaded: {df.shape}")

    # Feature engineering (optional)
    if args.feature_engineering:
        logger.info("Step 2: Applying feature engineering...")
        engineer = FeatureEngineer()
        df = engineer.feature_engineering_pipeline(df)
        logger.info(f"Features engineered: {df.shape}")
    else:
        logger.info("Step 2: Skipping feature engineering")

    # Preprocess data
    logger.info("Step 3: Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df,
        balance_method=args.balance_method,
        save_artifacts=True
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Train models
    logger.info("Step 4: Training models...")

    if args.hyperparameter_tuning:
        logger.info("Performing hyperparameter tuning...")
        trainer.get_models()
        # Tune selected models
        model_names = args.models if 'all' not in args.models else list(trainer.models.keys())
        for model_name in model_names:
            if model_name in trainer.models:
                logger.info(f"Tuning {model_name}...")
                tuned_model = trainer.hyperparameter_tuning(model_name, X_train, y_train)
                trainer.models[model_name] = tuned_model

    # Train and compare all models
    results = trainer.train_and_compare_models(X_train, y_train, X_test, y_test)

    # Save results
    logger.info("Step 5: Saving results...")
    trainer.save_model()
    trainer.save_results(results)

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Training Summary")
    logger.info("="*50)
    logger.info(f"Best Model: {trainer.best_model_name}")

    best_results = results[trainer.best_model_name]
    logger.info("\nTest Metrics:")
    for metric, value in best_results['test_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info(f"\nModel saved to: {config.BEST_MODEL_FILE}")
    logger.info(f"Results saved to: {config.REPORTS_DIR / 'metrics' / 'training_results.json'}")

    logger.info("\n" + "="*50)
    logger.info("Training completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)
