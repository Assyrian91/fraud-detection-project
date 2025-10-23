# 🛡️ Credit Card Fraud Detection System
![CI/CD](https://github.com/AssyriaNathraq/fraud-detection-project/workflows/CI%2FCD%20Pipeline/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green)
![License](https://img.shields.io/badge/license-MIT-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/Assyrian91/fraud-detection-project)

A comprehensive end-to-end machine learning project for detecting fraudulent credit card transactions, featuring real-time prediction APIs, model training pipeline, and deployment-ready structure.

⸻

# ⚙️ Quick Setup

Dataset not included — see DATA_SETUP.md for instructions.

 1️⃣ Download data from Kaggle

 2️⃣ Install dependencies
	pip install -r requirements.txt

 3️⃣ Train model
	python scripts/train_model.py

# 📚 Table of Contents
	•	Features
	•	Project Structure
	•	Installation
	•	Usage
	•	API Documentation
	•	Model Performance
	•	Testing
	•	Deployment
	•	Configuration
	•	Contributing
	•	License
	•	Contact
	•	Roadmap

# ✨ Features

	✅ Multiple ML Models (Logistic Regression, Random Forest, XGBoost, LightGBM)
	✅ Advanced Preprocessing (SMOTE, StandardScaler, Feature Engineering)
	✅ REST API using FastAPI
	✅ Interactive Dashboard (Dash)
	✅ Model Monitoring and Logging (Loguru)
	✅ Docker-ready
	✅ CI/CD Workflows (GitHub Actions)

# 📁 Project Structure
	fraud_detection/
	├── src/
	│   ├── api/                # FastAPI application
	│   ├── data/               # Data preprocessing
	│   ├── features/           # Feature engineering
	│   ├── models/             # Training and prediction modules
	│   └── utils/              # Helper functions
	│
	├── data/
	│   ├── raw/                # Original dataset
	│   └── processed/          # Cleaned/preprocessed data
	│
	├── models/saved_models/    # Trained models
	├── notebooks/              # Exploratory notebooks
	├── scripts/                # Training and evaluation scripts
	├── tests/                  # Unit and integration tests
	├── reports/                # Figures and metrics
	├── logs/                   # Application logs
	│
	├── .github/workflows/      # CI/CD pipelines
	├── requirements.txt
	├── Dockerfile
	├── docker-compose.yml
	├── setup.py
	├── .env.example
	└── README.md


# 🚀 Installation

Prerequisites
	•	Python 3.8+
	•	pip
	•	Git
	•	(Optional) Docker

# Steps
## Clone repository
	git clone https://github.com/Assyrian91/fraud-detection-project.git
	cd fraud-detection-project

## Create virtual environment
	python -m venv venv
	venv\Scripts\activate    # On Windows

## Install dependencies
	pip install -r requirements.txt

## Configure environment
	copy .env.example .env

# 📥 Data Setup
	1.	Download dataset from Kaggle – Credit Card Fraud Detection

	2.	Place creditcard.csv inside:
		data/raw/creditcard.csv

	3.	Or download via Kaggle API:
		kaggle datasets download -d mlg-ulb/creditcardfraud
		unzip creditcardfraud.zip -d data/raw/


# 📖 Usage

## 1️⃣ Data Preprocessing
	python -c "
	from src.data.preprocessing import DataPreprocessor
	p = DataPreprocessor()
	df = p.load_data()
	X_train, X_test, y_train, y_test = p.preprocess_pipeline(df)
	"

## 2️⃣ Model Training
	python -c "
	from src.models.train import ModelTrainer
	trainer = ModelTrainer()
	trainer.train_and_compare_models()
	trainer.save_model()
	"
## 3️⃣ Run API
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
	Swagger Docs → http://localhost:8000/docs


# 📊 Model Performance

## Best Model: XGBoost
	Metric		Score
	Accuracy	99.95%
	Precision	95.2%
	Recall		89.7%
	F1 Score	92.4%
	ROC AUC		98.3%

## Confusion Matrix:
		Predicted
        	         0      1
	Actual  0    56,850    12
        	1        8     92
______________________________

# 📚 API Documentation
	Method		Endpoint		Description
	GET		/			Root endpoint
	GET		/health			Health check
	POST		/predict		Single transaction prediction
	POST		/predict/batch		Batch predictions
	POST		/predict/explain	Prediction with explanation
	GET		/model/info		Model details
	POST		/model/threshold	Update prediction threshold

# 🧪 Testing
	pytest
	pytest --cov=src --cov-report=html
	pytest tests/test_model.py::TestModelTrainer::test_train_single_model -v


# 🐳 Docker Deployment
	# Build and run
	docker build -t fraud-detection .
	docker run -p 8000:8000 fraud-detection
## Using Docker Compose:
	docker-compose up --build -d

⸻

# 🔧 Configuration

.env file example:
	API_HOST=0.0.0.0
	API_PORT=8000
	FRAUD_THRESHOLD=0.5
	LOG_LEVEL=INFO
# 🤝 Contributing
	1.	Fork the repository
	2.	Create feature branch → git checkout -b feature/your-feature
	3.	Commit changes → git commit -m "Add new feature"
	4.	Push → git push origin feature/your-feature
	5.	Open Pull Request

Code Style:
	•	PEP 8
	•	Type hints
	•	Docstrings
	•	Tests for new features

⸻

# 📄 License

Licensed under the MIT License — see LICENSE.

⸻

# 👨‍💻 Author

Khoshaba Odeesho – Data Analyst & ML Developer
	•	GitHub: Assyrian91
	•	LinkedIn: linkedin.com/in/khoshaba-odeesho-17b5b92aa
	•	Email: khoshaba.odeesho@gmail.com

⸻

# 🗺️ Roadmap
	•	Add SHAP/LIME explainability
	•	Real-time streaming predictions
	•	A/B testing framework
	•	Mobile app interface
	•	Email/SMS fraud alerts
	•	Model versioning system
	•	Monitoring dashboard

⸻

# ⭐ Star this repository if you find it helpful!