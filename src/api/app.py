"""
FastAPI application for fraud detection
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from loguru import logger
import sys

from src.config import Config
from src.models.predict import FraudPredictor

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(Config.LOG_FILE, rotation="1 day", retention="7 days", level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for credit card fraud detection using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    predictor = FraudPredictor()
    logger.info("Fraud predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

# Pydantic models
class Transaction(BaseModel):
    """Single transaction model"""
    Time: Optional[float] = Field(default=0.0, description="Time in seconds from first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount")

    class Config:
        schema_extra = {
            "example": {
                "Time": 0,
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
            }
        }

class BatchTransactions(BaseModel):
    """Batch of transactions"""
    transactions: List[Transaction] = Field(..., min_items=1, max_items=1000)

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    transaction_id: int
    is_fraud: bool
    fraud_probability: float
    confidence: float
    risk_level: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    total_transactions: int
    fraud_count: int
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str

# API Endpoints
@app.get("/", tags=["General"])
async def root():
    return {"message": "Fraud Detection API", "version": "1.0.0", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single_transaction(transaction: Transaction):
    """Predict fraud for a single transaction"""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    try:
        transaction_dict = transaction.dict()
        result = predictor.predict_with_confidence(transaction_dict)[0]
        logger.info(f"Prediction made: {result}")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_transactions(batch: BatchTransactions):
    """Predict fraud for multiple transactions"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        transactions_list = [t.dict() for t in batch.transactions]
        results = predictor.predict_with_confidence(transactions_list)
        fraud_count = sum(1 for r in results if r['is_fraud'])
        response = {
            "total_transactions": len(results),
            "fraud_count": fraud_count,
            "predictions": results
        }
        logger.info(f"Batch prediction: {len(results)} transactions, {fraud_count} frauds detected")
        return response
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/explain", tags=["Prediction"])
async def explain_prediction(transaction: Transaction):
    """Get detailed explanation for a prediction"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        transaction_dict = transaction.dict()
        explanation = predictor.explain_prediction(transaction_dict)
        logger.info("Prediction explanation generated")
        return explanation
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": type(predictor.model).__name__,
        "threshold": predictor.threshold,
        "features_count": len(Config.FEATURE_COLUMNS)
    }

@app.post("/model/threshold", tags=["Model"])
async def update_threshold(new_threshold: float):
    """Update the prediction threshold"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not 0 <= new_threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")
    try:
        old_threshold = predictor.threshold
        predictor.set_threshold(new_threshold)
        logger.info(f"Threshold updated: {old_threshold} -> {new_threshold}")
        return {
            "message": "Threshold updated successfully",
            "old_threshold": old_threshold,
            "new_threshold": new_threshold
        }
    except Exception as e:
        logger.error(f"Threshold update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update threshold: {str(e)}")

if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        "src.api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level="info"
    )
