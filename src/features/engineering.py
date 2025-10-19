"""
Feature Engineering module for fraud detection
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from loguru import logger

from src.config import Config

class FeatureEngineer:
    """Class for feature engineering operations"""

    def __init__(self):
        self.config = Config()
        self.pca = None
        self.feature_selector = None

    def create_time_features(self, df):
        """Create time-based features from Time column"""
        logger.info("Creating time-based features...")
        df = df.copy()
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Time_Period'] = pd.cut(
            df['Hour'], 
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        time_dummies = pd.get_dummies(df['Time_Period'], prefix='Period')
        df = pd.concat([df, time_dummies], axis=1)
        df = df.drop('Time_Period', axis=1)
        logger.info(f"Created time features. New shape: {df.shape}")
        return df

    def create_amount_features(self, df):
        """Create amount-based features"""
        logger.info("Creating amount-based features...")
        df = df.copy()
        df['Amount_Log'] = np.log1p(df['Amount'])
        df['Amount_Category'] = pd.cut(
            df['Amount'],
            bins=[0, 10, 50, 100, 500, np.inf],
            labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        )
        amount_dummies = pd.get_dummies(df['Amount_Category'], prefix='Amount')
        df = pd.concat([df, amount_dummies], axis=1)
        df = df.drop('Amount_Category', axis=1)
        logger.info(f"Created amount features. New shape: {df.shape}")
        return df

    def create_interaction_features(self, df):
        """Create interaction features between V columns"""
        logger.info("Creating interaction features...")
        df = df.copy()
        important_v_features = ['V1', 'V3', 'V4', 'V10', 'V12', 'V14', 'V17']
        for i, col1 in enumerate(important_v_features):
            for col2 in important_v_features[i+1:]:
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        logger.info(f"Created interaction features. New shape: {df.shape}")
        return df

    def create_statistical_features(self, df):
        """Create statistical features from V columns"""
        logger.info("Creating statistical features...")
        df = df.copy()
        v_columns = [col for col in df.columns if col.startswith('V')]
        df['V_mean'] = df[v_columns].mean(axis=1)
        df['V_std'] = df[v_columns].std(axis=1)
        df['V_min'] = df[v_columns].min(axis=1)
        df['V_max'] = df[v_columns].max(axis=1)
        df['V_range'] = df['V_max'] - df['V_min']
        df['V_median'] = df[v_columns].median(axis=1)
        df['V_sum_abs'] = df[v_columns].abs().sum(axis=1)
        logger.info(f"Created statistical features. New shape: {df.shape}")
        return df

    def apply_pca(self, X_train, X_test, n_components=10):
        """Apply PCA for dimensionality reduction"""
        logger.info(f"Applying PCA with {n_components} components...")
        v_columns = [col for col in X_train.columns if col.startswith('V')]
        self.pca = PCA(n_components=n_components, random_state=self.config.RANDOM_STATE)
        X_train_pca = self.pca.fit_transform(X_train[v_columns])
        X_test_pca = self.pca.transform(X_test[v_columns])
        pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
        X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)
        X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns, index=X_test.index)
        non_v_columns = [col for col in X_train.columns if not col.startswith('V')]
        X_train_final = pd.concat([X_train[non_v_columns].reset_index(drop=True), 
                                   X_train_pca_df.reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([X_test[non_v_columns].reset_index(drop=True), 
                                  X_test_pca_df.reset_index(drop=True)], axis=1)
        explained_var = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA completed. Explained variance: {explained_var:.4f}")
        return X_train_final, X_test_final

    def select_features(self, X_train, y_train, X_test, k=20, method='f_classif'):
        """Select top k features using statistical tests"""
        logger.info(f"Selecting top {k} features using {method}...")
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            logger.warning(f"Unknown method: {method}. Using f_classif.")
            score_func = f_classif
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = X_train.columns[selected_indices].tolist()
        logger.info(f"Selected features: {selected_features}")
        X_train_df = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        return X_train_df, X_test_df, selected_features

    def feature_engineering_pipeline(self, df, apply_pca_flag=False, select_features_flag=False):
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_statistical_features(df)
        df = self.create_interaction_features(df)
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df


if __name__ == "__main__":
    # Example usage
    from data.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()

    df = preprocessor.load_data()
    df_engineered = engineer.feature_engineering_pipeline(df)
