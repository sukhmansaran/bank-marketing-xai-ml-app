# Models Directory

## Overview

This directory contains trained machine learning models and their preprocessors.

## Files

### Random Forest Model
- **model_rf.pkl**: Trained Random Forest Classifier
- **preprocessors_rf.pkl**: Preprocessing objects (encoders, scalers)

### XGBoost Model (Optional)
- **model_xgb.pkl**: Trained XGBoost Classifier
- **preprocessors_xgb.pkl**: Preprocessing objects

## Model Generation

### Train Random Forest
```bash
python src/train_model.py
```

Generates:
- `models/model_rf.pkl` (Random Forest model)
- `models/preprocessors_rf.pkl` (Preprocessing pipeline)

### Train XGBoost
```bash
python src/train_model_xg.py
```

Generates:
- `models/model_xgb.pkl` (XGBoost model)
- `models/preprocessors_xgb.pkl` (Preprocessing pipeline)

## Model Performance

### Random Forest
- Accuracy: 87.3%
- Precision: 0.78
- Recall: 0.52
- F1-Score: 0.62
- ROC-AUC: 0.78

### XGBoost
- Accuracy: 88.1%
- Precision: 0.81
- Recall: 0.55
- F1-Score: 0.65
- ROC-AUC: 0.80

See [docs/EVALUATION.md](../docs/EVALUATION.md) for detailed evaluation report.

## Model Usage

### Loading Models
```python
import pickle

# Load Random Forest model
with open('models/model_rf.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessors
with open('models/preprocessors_rf.pkl', 'rb') as f:
    preprocessors = pickle.load(f)
```

### Making Predictions
```python
import pandas as pd

# Prepare input data
input_data = pd.DataFrame({
    'age': [35],
    'job': ['technician'],
    # ... other features
})

# Preprocess
le_dict = preprocessors['label_encoders']
scaler = preprocessors['scaler']

# Make prediction
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)
```

## Model Specifications

### Random Forest
- **Algorithm**: RandomForestClassifier
- **n_estimators**: 300
- **max_depth**: 15
- **random_state**: 42
- **n_jobs**: -1 (parallel processing)

### XGBoost
- **Algorithm**: XGBClassifier
- **n_estimators**: 300
- **max_depth**: 12
- **learning_rate**: 0.05
- **min_child_weight**: 5

## Preprocessing Pipeline

### Categorical Features
- LabelEncoder for each categorical column
- Stored in `preprocessors['label_encoders']`

### Numeric Features
- StandardScaler for normalization
- Stored in `preprocessors['scaler']`

### Feature Lists
- `preprocessors['numeric_columns']`: List of numeric features
- `preprocessors['non_numeric_columns']`: List of categorical features

## Model Deployment

### Streamlit App
```bash
streamlit run app/streamlit_app.py
```

The app automatically loads the best available model (RF or XGB).

### Docker
```bash
docker build -t bank-deposit-xai .
docker run -p 8501:8501 bank-deposit-xai
```

### API (FastAPI - Optional)
```bash
python api/main.py
```

## Model Monitoring

### Performance Tracking
- Monitor accuracy on new data
- Track prediction distribution
- Check for model drift

### Retraining
- Retrain monthly or when performance drops
- Use cross-validation for stability
- Compare with baseline models

## Model Versioning

### Current Version
- **Random Forest**: v1.0
- **XGBoost**: v1.0

### Version History
- v1.0: Initial models with 87-88% accuracy

### Future Versions
- v1.1: Improved feature engineering
- v2.0: Ensemble methods
- v2.1: Hyperparameter optimization

## Notes

- Models are regenerated when training scripts are run
- Pickle files are not version controlled (.gitignore)
- Always validate models before deployment
- Keep preprocessors in sync with models
- Document any model changes

## Troubleshooting

### Model Not Found
```bash
# Retrain the model
python src/train_model.py
```

### Preprocessing Error
```bash
# Ensure preprocessors are loaded correctly
import pickle
with open('models/preprocessors_rf.pkl', 'rb') as f:
    preprocessors = pickle.load(f)
print(preprocessors.keys())
```

### Prediction Mismatch
- Verify input data format
- Check feature order matches training
- Ensure categorical values are in training set

## References

- [Scikit-learn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Model Evaluation](../docs/EVALUATION.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)
