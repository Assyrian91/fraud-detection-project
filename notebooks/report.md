# Fraud Detection Pipeline

## Overview
This project implements a full MLOps pipeline to predict customer churn.  
It includes data preprocessing, feature engineering, model training, evaluation, version control, and API deployment.  
The pipeline is modular, reproducible, and production-ready.

---

## 1. Data Preparation

- Raw data stored in `data/raw/`
- Cleaned and preprocessed using `pandas` and saved to `data/processed/`
- Missing values handled, categorical features encoded, and outliers removed
- Versioned using DVC for reproducibility

```bash
dvc add data/processed/clean.csv


---

2. Feature Engineering

• Selected relevant features based on domain knowledge and correlation analysis
• Created derived features (e.g., tenure buckets, contract type encoding)
• Feature set stored in `data/features/`


---

3. Model Training

• Model: `RandomForestClassifier` from `scikit-learn`
• Training script: `src/train.py`
• Parameters defined in `params.yaml`
• Metrics logged to `metrics.json`


python src/train.py


---

4. Evaluation

• Accuracy: 0.87
• Classification report includes precision, recall, and F1-score
• Confusion matrix plotted and saved to `notebooks/`


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))


---

5. Experiment Tracking

• Pipeline stages defined in `dvc.yaml`
• Lock file: `dvc.lock` ensures reproducibility
• Metrics tracked via DVC and stored in `metrics.json`


dvc repro
dvc metrics show


---

6. API Deployment

• Framework: FastAPI
• Entry point: `src/api/app.py`
• Served using Uvicorn


uvicorn src.api.app:app --host 0.0.0.0 --port 8000


---

7. API Endpoints

Endpoint	Method	Description	
/ping	GET	Health check	
/predict	POST	Returns churn prediction	
/docs	GET	Swagger documentation	


Example request:

{
  "customer_id": "C12345",
  "features": {
    "tenure": 12,
    "monthly_charges": 75.5,
    "contract_type": "One year"
  }
}


---

8. CI/CD & Code Quality

• Pre-commit hooks: `black`, `flake8`, `isort`
• GitHub Actions for automated testing and deployment
• Folder structure cleaned and modularized


pre-commit run --all-files


---

9. Deployment Infrastructure

• Cloud: AWS EC2 (Ubuntu)
• Storage: AWS S3 (optional for model/data artifacts)
• Service manager: `systemd` for persistent API hosting
• SSH and security groups configured manually


---

10. Future Improvements

• Add model explainability via SHAP
• Integrate monitoring tools (e.g., Prometheus + Grafana)
• Enable batch prediction and logging
• Automate retraining pipeline with cron jobs


---

References

• `requirements.txt`: all dependencies
• `dvc.yaml`: pipeline stages
• `params.yaml`: model parameters
• `metrics.json`: evaluation results
• `src/`: source code
• `notebooks/`: analysis and documentation
