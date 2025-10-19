"""
Configuration file for the fraud detection system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the project"""

    # Project Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    REPORTS_DIR = BASE_DIR / "reports"

    # Data Files
    RAW_DATA_FILE = RAW_DATA_DIR / "creditcard.csv"
    CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_creditcard.csv"
    TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_data.csv"
    TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_data.csv"

    # Model Files
    MODEL_FILE = MODELS_DIR / "logistic_model.pkl"
    SCALER_FILE = MODELS_DIR / "scaler.pkl"
    BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"

    # Model Parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

    # Class Imbalance Handling
    SAMPLING_STRATEGY = "auto"  # for SMOTE

    # Model Training
    MAX_ITER = 1000
    N_JOBS = -1  # Use all CPU cores

    # Threshold for classification
    FRAUD_THRESHOLD = 0.5

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "fraud_detection.log"

    # Dashboard
    DASH_HOST = os.getenv("DASH_HOST", "0.0.0.0")
    DASH_PORT = int(os.getenv("DASH_PORT", 8050))
    DASH_DEBUG = os.getenv("DASH_DEBUG", "True").lower() == "true"

    # Feature Engineering
    FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    TARGET_COLUMN = "Class"

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.REPORTS_DIR,
            cls.REPORTS_DIR / "figures",
            cls.REPORTS_DIR / "metrics"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_paths(cls):
        """Validate that required files exist"""
        if not cls.RAW_DATA_FILE.exists():
            raise FileNotFoundError(f"Raw data file not found: {cls.RAW_DATA_FILE}")
        return True

# Create directories on import
Config.create_directories()
