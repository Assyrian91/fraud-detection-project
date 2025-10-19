# 🛡️ Credit Card Fraud Detection Project

### ⚠️ Important: Data Setup Required

The dataset is **not included** in this repository. Please see [DATA_SETUP.md](DATA_SETUP.md) for download instructions.

**Quick Start:**
```bash
# 1. Download data from Kaggle
# See DATA_SETUP.md

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python scripts/train_model.py

A comprehensive machine learning system for detecting fraudulent credit card transactions using advanced ML techniques and real-time prediction API.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting
- **Advanced Preprocessing**: SMOTE for imbalanced data, StandardScaler, feature engineering
- **REST API**: FastAPI-based API with comprehensive endpoints
- **Interactive Dashboard**: Real-time visualization using Dash
- **Model Monitoring**: Cross-validation, hyperparameter tuning, performance metrics
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Comprehensive Testing**: Unit tests and integration tests
- **Logging**: Structured logging with Loguru
- **Configuration Management**: Environment-based configuration

## 📁 Project Structure

```
fraud_detection/
├── .github/
│   └── workflows/          # CI/CD pipelines
│       ├── tests.yml
│       └── deploy.yml
├── data/
│   ├── raw/               # Original dataset
│   ├── processed/         # Cleaned and preprocessed data
│   └── external/          # External data sources
├── models/
│   └── saved_models/      # Trained model files
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── reports/
│   ├── figures/           # Plots and visualizations
│   └── metrics/           # Performance metrics
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py         # FastAPI application
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── logs/
├── scripts/
│   ├── train_model.py
│   └── evaluate_model.py
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Git
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository**

```bash
git clone https://github.com/Assyrian91/fraud-detection-project.git
```

1. **Create virtual environment**

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fraud_detection python=3.10
conda activate fraud_detection
```

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

1. **Setup environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

1. **Download dataset**

Place the `creditcard.csv` dataset in `data/raw/` directory.

## ⚡ Quick Start

### 1. Data Preprocessing

```bash
# Run preprocessing pipeline
python -c "
from src.data.preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
df = preprocessor.load_data()
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df)
"
```
## 📥 Data Setup

The dataset is not included in this repository due to its size.

### Download Dataset:

1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in `data/raw/creditcard.csv`

### Alternative:

```bash
# Using Kaggle API
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/
### 2. Train Models

```bash
# Train and compare multiple models
python -c "
from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer

# Load preprocessed data
preprocessor = DataPreprocessor()
df = preprocessor.load_data()
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df)

# Train models
trainer = ModelTrainer()
results = trainer.train_and_compare_models(X_train, y_train, X_test, y_test)
trainer.save_model()
"
```

### 3. Start API Server

```bash
# Development mode
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Start Dashboard

```bash
python dash_interface.py
```

Access the dashboard at `http://localhost:8050`

## 📖 Usage

### Using Python API

```python
from src.models.predict import FraudPredictor

# Initialize predictor
predictor = FraudPredictor()

# Single transaction
transaction = {
    'V1': -1.359807,
    'V2': -0.072781,
    'V3': 2.536347,
    # ... other features
    'Amount': 149.62
}

# Get prediction with confidence
result = predictor.predict_with_confidence(transaction)
print(result)
# Output: {
#   'transaction_id': 0,
#   'is_fraud': True,
#   'fraud_probability': 0.85,
#   'confidence': 0.70,
#   'risk_level': 'High'
# }
```

### Using REST API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {...},  # Transaction 1
      {...}   # Transaction 2
    ]
  }'
```

## 📚 API Documentation

Once the API is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Main Endpoints

|Method|Endpoint          |Description                  |
|------|------------------|-----------------------------|
|GET   |`/`               |Root endpoint                |
|GET   |`/health`         |Health check                 |
|POST  |`/predict`        |Single transaction prediction|
|POST  |`/predict/batch`  |Batch predictions            |
|POST  |`/predict/explain`|Prediction with explanation  |
|GET   |`/model/info`     |Model information            |
|POST  |`/model/threshold`|Update prediction threshold  |

## 📊 Model Performance

### Best Model: XGBoost

|Metric   |Score |
|---------|------|
|Accuracy |99.95%|
|Precision|95.2% |
|Recall   |89.7% |
|F1 Score |92.4% |
|ROC AUC  |98.3% |

### Confusion Matrix

```
                 Predicted
                 0      1
Actual  0    56,850    12
        1        8     92
```

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run specific test
pytest tests/test_model.py::TestModelTrainer::test_train_single_model -v
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing
```

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build image
docker build -t fraud-detection .

# Run container
docker run -p 8000:8000 fraud-detection
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up --build -d
```

Services:

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8050`

## 🔧 Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Model Configuration
FRAUD_THRESHOLD=0.5

# Logging
LOG_LEVEL=INFO
```

### Model Configuration

Edit `src/config.py` to customize:

- Data paths
- Model parameters
- Feature engineering settings
- Cross-validation settings

## 📈 Feature Engineering

The system includes advanced feature engineering:

1. **Time-based features**
- Hour of day
- Time periods (morning, afternoon, evening, night)
1. **Amount-based features**
- Log transformation
- Amount categories
- Statistical aggregations
1. **Interaction features**
- Pairwise interactions between important V features
1. **Statistical features**
- Mean, std, min, max, range of V features

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflows:

- **Tests**: Run on every push/PR
- **Code Quality**: Linting and formatting checks
- **Deployment**: Automated deployment to production

## 📝 Best Practices

1. **Data Preprocessing**
- Always scale features
- Handle class imbalance with SMOTE
- Use stratified splits
1. **Model Training**
- Use cross-validation
- Perform hyperparameter tuning
- Compare multiple models
1. **Deployment**
- Monitor model performance
- Log predictions
- Set up alerts for anomalies
1. **Security**
- Never commit sensitive data
- Use environment variables
- Implement rate limiting

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/AmazingFeature`)
1. Commit your changes (`git commit -m 'Add AmazingFeature'`)
1. Push to the branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the <LICENSE> file for details.

## 👥 Authors

- - **Khoshaba Odeesho** - *Data Analyst & ML Developer* - [GitHub](https://github.com/Assyrian91)

## 🙏 Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Scikit-learn team
- FastAPI developers
- Dash Plotly team

## 📞 Contact

For questions or support:

- Email: khoshaba.odeesho@gmail.com
- LinkedIn: [linkedin.com/in/khoshaba-odeesho-17b5b92aa](http://linkedin.com/in/khoshaba-odeesho-17b5b92aa)

## 🗺️ Roadmap

- [ ] Add SHAP/LIME explainability
- [ ] Implement real-time streaming predictions
- [ ] Add A/B testing framework
- [ ] Create mobile app
- [ ] Add email/SMS alerts
- [ ] Implement model versioning
- [ ] Add performance monitoring dashboard

-----

**⭐ Star this repo if you find it helpful!**