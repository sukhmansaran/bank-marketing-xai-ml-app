# Bank Marketing XAI ML App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sukhmansaran-bank-marketing-xai-ml-app-appstreamlit-app-6kguh6.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready machine learning application predicting bank customer term deposit subscriptions using XGBoost and SHAP for complete model interpretability.

## 🚀 Live Demo

**Try it now**: [https://sukhmansaran-bank-marketing-xai-ml-app-appstreamlit-app-2nypva.streamlit.app/](https://sukhmansaran-bank-marketing-xai-ml-app-appstreamlit-app-2nypva.streamlit.app/)

> Deployed on Streamlit Cloud - No installation required!

## Key Results

- **Accuracy**: 87-88%
- **ROC-AUC**: 0.78-0.80
- **Model**: XGBoost Classifier with GridSearchCV tuning
- **Explainability**: SHAP global + local explanations
- **Deployment**: Docker + Streamlit web app
- **Tech Stack**: XGBoost, SHAP, Streamlit, Scikit-learn

## System Architecture

```
User Input → Streamlit UI → Preprocessing (Label Encoders) → XGBoost Model → Prediction
                                                                    ↓
                                                              SHAP Explainer
                                                                    ↓
                                                    Visualizations (Force Plot, Summary, Dependence)
```

**Pipeline Flow:**
1. User enters customer data via Streamlit interface
2. Categorical features encoded using saved LabelEncoders
3. XGBoost model generates prediction + probability
4. SHAP TreeExplainer computes feature contributions
5. Interactive visualizations explain the prediction

## Project Structure

```
bank-deposit-xai-app/
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── src/
│   └── train_model_xg.py         # XGBoost training script
├── data/
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed datasets (train/test)
├── models/                       # Trained models & encoders
│   ├── model_xgb.json           # XGBoost model (JSON format)
│   └── label_encoders_xgb.pkl   # Label encoders for categorical features
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose setup
└── README.md                     # This file
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/bank-marketing-xai-ml-app.git
cd bank-marketing-xai-ml-app
```

### 2. Download Dataset
Download the Bank Marketing dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and place CSV files in `data/raw/`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python src/train_model_xg.py
```

This generates:
- `models/model_xgb.json` - XGBoost model
- `models/label_encoders_xgb.pkl` - Categorical feature encoders

### 5. Run the App
```bash
streamlit run app/streamlit_app.py
```

Open browser to `http://localhost:8501`

## Features

### 📊 Home
Overview and introduction to the application

### 📈 Data Analysis
- Explore training and test datasets
- View dataset statistics and distributions
- Check for missing values

### 🔮 Make Prediction
- Interactive form to input customer details
- Instant predictions with confidence scores
- **SHAP Force Plot**: Visual explanation of why the model made this prediction
- **Feature Contributions Table**: Detailed breakdown of each feature's impact

### 🔍 Model Interpretability
Comprehensive SHAP-based model explanations:
- **Summary Plot (Bar)**: Feature importance rankings
- **Summary Plot (Beeswarm)**: How feature values influence predictions
- **Dependence Plots**: Relationship between individual features and predictions

### 📋 Model Info
- Model type and training status
- Feature importance visualization
- Top 15 most influential features

> **Note**: Add screenshots to `images/` directory for better visual documentation. Recommended screenshots: prediction interface, SHAP force plot, summary plots.

## Dataset

Portuguese bank marketing campaign data with 20 features + 1 target:

**Customer Info**: age, job, marital, education, default, housing, loan  
**Campaign Details**: contact, month, day_of_week, duration, campaign, pdays, previous, poutcome  
**Economic Indicators**: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

**Target**: Binary classification (will/won't subscribe to term deposit)

**Data Source**: UCI Machine Learning Repository - Bank Marketing Dataset

**Note**: Raw datasets are not included in this repository. Download from:
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Place files in `data/raw/` directory before training

## Model Details

- **Algorithm**: XGBoost Classifier
- **Performance**: 87-88% accuracy, 0.78-0.80 ROC-AUC
- **Hyperparameter Tuning**: GridSearchCV with F1 scoring (max_depth, n_estimators, learning_rate, min_child_weight)
- **Preprocessing**: LabelEncoder for categorical features (SHAP-compatible)
- **Model Format**: JSON (enables categorical feature support)

## Understanding SHAP Explanations

SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions:

- **Red**: Pushes prediction toward "Will Subscribe"
- **Blue**: Pushes prediction toward "Won't Subscribe"
- **Magnitude**: Larger values = stronger influence

### Use Cases
- Understand why a specific customer is predicted to subscribe
- Identify which features drive model decisions globally
- Detect potential biases or unexpected patterns
- Build trust in model predictions

## Why This Project Stands Out

**Production-Ready Architecture**
- Clean separation: UI (Streamlit) → Logic (XGBoost) → Explainability (SHAP)
- Proper preprocessing pipeline with saved encoders
- Docker containerization for consistent deployment

**Explainable AI Integration**
- Not just predictions - full transparency via SHAP
- Local explanations (per-prediction force plots)
- Global explanations (feature importance, summary plots)

**Engineering Best Practices**
- Modular code structure
- Comprehensive documentation
- Version control ready (.gitignore configured)
- Reproducible training pipeline

## Technical Stack

- **Streamlit**: Interactive web interface
- **XGBoost**: Gradient boosting classifier
- **SHAP**: Model interpretability & explanations
- **Scikit-learn**: Preprocessing & evaluation
- **Pandas**: Data manipulation
- **Matplotlib**: Visualizations

## Docker Deployment

### Live Demo
The app is deployed on Streamlit Cloud: [Live Demo](https://sukhmansaran-bank-marketing-xai-ml-app-appstreamlit-app-6kguh6.streamlit.app/)

### Build and Run Locally
```bash
# Build image
docker build -t bank-deposit-xai:latest .

# Run container
docker run -p 8501:8501 bank-deposit-xai:latest
```

### Using Docker Compose
```bash
docker-compose up -d
```

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed system architecture & design decisions
- **[EVALUATION.md](docs/EVALUATION.md)** - Model evaluation & metrics
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide
- **[GIT_GUIDELINES.md](docs/GIT_GUIDELINES.md)** - Git workflow
- **[PORTFOLIO_IMPROVEMENTS.md](docs/PORTFOLIO_IMPROVEMENTS.md)** - Enhancement roadmap
- **[PORTFOLIO_CHECKLIST.md](docs/PORTFOLIO_CHECKLIST.md)** - Pre-upload checklist

## Notes

- Train model before making predictions: `python src/train_model_xg.py`
- Label encoders required for SHAP compatibility (categorical features → integers)
- SHAP calculations cached for performance
- Model saved as JSON format for XGBoost compatibility
- See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment

## Contributing

See [GIT_GUIDELINES.md](docs/GIT_GUIDELINES.md) for contribution guidelines and workflow.

## Repository Naming

For better discoverability and professional presentation, consider renaming to:
- `bank-marketing-xai-ml-app` (recommended)
- `bank-deposit-prediction-xai-streamlit`
- `xgboost-shap-banking-ml`

Clear, descriptive names improve GitHub searchability and recruiter perception.

## License

This project is open source and available under the MIT License.
