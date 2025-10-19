"""
Data preprocessing module for fraud detection
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from loguru import logger

from src.config import Config

class DataPreprocessor:
    """Class for data preprocessing operations"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.config = Config()

    def load_data(self, filepath=None):
        """Load data from CSV file"""
        if filepath is None:
            filepath = self.config.RAW_DATA_FILE

        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def clean_data(self, df):
        """Clean the dataset"""
        logger.info("Starting data cleaning...")

        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")

        # Handle missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
            df = df.dropna()
            logger.info("Dropped rows with missing values")

        # Remove outliers in Amount column (optional)
        Q1 = df['Amount'].quantile(0.25)
        Q3 = df['Amount'].quantile(0.75)
        IQR = Q3 - Q1

        fraud_before = df[df['Class'] == 1].shape[0]

        outlier_mask = (df['Amount'] < Q1 - 3 * IQR) | (df['Amount'] > Q3 + 3 * IQR)
        df_cleaned = df[~outlier_mask]

        fraud_after = df_cleaned[df_cleaned['Class'] == 1].shape[0]
        logger.info(f"Removed {df.shape[0] - df_cleaned.shape[0]} extreme outliers")
        logger.info(f"Fraud cases: {fraud_before} -> {fraud_after}")

        return df_cleaned

    def split_data(self, df, test_size=None, random_state=None):
        """Split data into train and test sets"""
        if test_size is None:
            test_size = self.config.TEST_SIZE
        if random_state is None:
            random_state = self.config.RANDOM_STATE

        logger.info(f"Splitting data with test_size={test_size}")

        X = df.drop(self.config.TARGET_COLUMN, axis=1)
        y = df[self.config.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        logger.info(f"Train fraud ratio: {y_train.sum() / len(y_train):.4f}")
        logger.info(f"Test fraud ratio: {y_test.sum() / len(y_test):.4f}")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test, fit=True):
        """Scale features using StandardScaler"""
        logger.info("Scaling features...")
        cols_to_scale = [col for col in X_train.columns if col not in ['Time']]

        if fit:
            X_train_scaled = X_train.copy()
            X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
            logger.info("Scaler fitted and applied to training data")
        else:
            X_train_scaled = X_train.copy()
            X_train_scaled[cols_to_scale] = self.scaler.transform(X_train[cols_to_scale])

        X_test_scaled = X_test.copy()
        X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])

        return X_train_scaled, X_test_scaled

    def handle_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance using SMOTE or other methods"""
        logger.info(f"Handling class imbalance using {method.upper()}...")

        initial_fraud = y_train.sum()
        initial_total = len(y_train)
        logger.info(f"Before balancing - Fraud: {initial_fraud}, Total: {initial_total}, Ratio: {initial_fraud/initial_total:.4f}")

        if method.lower() == 'smote':
            over = SMOTE(sampling_strategy=0.5, random_state=self.config.RANDOM_STATE)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=self.config.RANDOM_STATE)
            pipeline = ImbPipeline([('over', over), ('under', under)])
            X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

        elif method.lower() == 'undersample':
            under = RandomUnderSampler(sampling_strategy=0.5, random_state=self.config.RANDOM_STATE)
            X_resampled, y_resampled = under.fit_resample(X_train, y_train)

        else:
            logger.warning(f"Unknown balancing method: {method}. Returning original data.")
            return X_train, y_train

        final_fraud = y_resampled.sum()
        final_total = len(y_resampled)
        logger.info(f"After balancing - Fraud: {final_fraud}, Total: {final_total}, Ratio: {final_fraud/final_total:.4f}")

        return X_resampled, y_resampled

    def save_scaler(self, filepath=None):
        """Save the fitted scaler"""
        if filepath is None:
            filepath = self.config.SCALER_FILE

        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath=None):
        """Load a fitted scaler"""
        if filepath is None:
            filepath = self.config.SCALER_FILE

        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")
        return self.scaler

    def preprocess_pipeline(self, df, balance_method='smote', save_artifacts=True):
        """Complete preprocessing pipeline"""
        logger.info("Starting complete preprocessing pipeline...")

        df_cleaned = self.clean_data(df)
        X_train, X_test, y_train, y_test = self.split_data(df_cleaned)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train_scaled, y_train, method=balance_method)

        if save_artifacts:
            self.save_scaler()
            train_df = pd.concat([X_train_balanced, y_train_balanced], axis=1)
            test_df = pd.concat([X_test_scaled, y_test], axis=1)
            train_df.to_csv(self.config.TRAIN_DATA_FILE, index=False)
            test_df.to_csv(self.config.TEST_DATA_FILE, index=False)
            logger.info("Processed data saved")

        logger.info("Preprocessing pipeline completed successfully!")

        return X_train_balanced, X_test_scaled, y_train_balanced, y_test


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df)
