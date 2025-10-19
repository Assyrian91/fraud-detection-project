# Data Directory

## Structure

data/
├── raw/           # Place creditcard.csv here
└── processed/     # Generated processed data files

## Download Dataset

Download the Credit Card Fraud Detection dataset from Kaggle:

**Kaggle Dataset**: https://www.kaggle.com/mlg-ulb/creditcardfraud

1. Download `creditcard.csv`
2. Place it in `data/raw/creditcard.csv`
3. Run preprocessing: `python scripts/train_model.py`

## Note

Large data files are not included in the repository. 
Please download the dataset separately.