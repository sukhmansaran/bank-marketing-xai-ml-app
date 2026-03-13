# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                      (Streamlit Web App)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT PREPROCESSING                          │
│  • Collect user input (20 features)                            │
│  • Apply LabelEncoders to categorical features                 │
│  • Convert to model-ready format                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      XGBOOST MODEL                              │
│  • Load model from JSON (models/model_xgb.json)                │
│  • Generate prediction (0 or 1)                                │
│  • Calculate probability scores                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SHAP EXPLAINER                              │
│  • TreeExplainer for XGBoost                                   │
│  • Compute SHAP values for each feature                        │
│  • Calculate base value and contributions                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATIONS                               │
│  • Force Plot: Individual prediction explanation               │
│  • Feature Contributions: Tabular breakdown                    │
│  • Summary Plots: Global feature importance                    │
│  • Dependence Plots: Feature relationships                     │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Streamlit UI Layer
- **Purpose**: User interaction and visualization
- **Features**: 
  - Interactive forms for data input
  - Real-time prediction display
  - Dynamic SHAP visualizations
  - Multi-page navigation
- **Files**: `app/streamlit_app.py`

### 2. Preprocessing Pipeline
- **Purpose**: Transform raw input to model format
- **Components**:
  - LabelEncoders for categorical features (job, education, marital, etc.)
  - Feature validation and type conversion
  - Consistent encoding with training data
- **Files**: `models/label_encoders_xgb.pkl`

### 3. XGBoost Model
- **Purpose**: Binary classification prediction
- **Specifications**:
  - Algorithm: Gradient Boosted Trees
  - Hyperparameters: Tuned via GridSearchCV
  - Format: JSON (for compatibility)
  - Performance: 87-88% accuracy, 0.78-0.80 ROC-AUC
- **Files**: `models/model_xgb.json`

### 4. SHAP Explainer
- **Purpose**: Model interpretability
- **Method**: TreeExplainer (optimized for tree-based models)
- **Outputs**:
  - SHAP values per feature
  - Base value (expected model output)
  - Feature contributions (positive/negative)
- **Library**: `shap>=0.42.0`

### 5. Visualization Layer
- **Purpose**: Present predictions and explanations
- **Visualizations**:
  - **Force Plot**: Waterfall showing feature push/pull
  - **Summary Plot (Bar)**: Mean absolute SHAP values
  - **Summary Plot (Beeswarm)**: Feature value impact distribution
  - **Dependence Plot**: Feature vs SHAP value relationship
- **Libraries**: `matplotlib`, `shap.plots`

## Data Flow

```
Raw Input (20 features)
    ↓
Label Encoding (categorical → integers)
    ↓
XGBoost Prediction (binary + probability)
    ↓
SHAP Computation (feature contributions)
    ↓
Visualization Rendering (plots + tables)
    ↓
Display to User
```

## Training Pipeline

```
Raw Data (data/raw/*.csv)
    ↓
Preprocessing (train/test split)
    ↓
Label Encoding (fit on training data)
    ↓
GridSearchCV (hyperparameter tuning)
    ↓
Model Training (XGBoost)
    ↓
Model Evaluation (accuracy, ROC-AUC, F1)
    ↓
Save Artifacts
    • model_xgb.json
    • label_encoders_xgb.pkl
```

## Deployment Architecture

### Local Development
```
Developer Machine
    ↓
Python Virtual Environment
    ↓
Streamlit Dev Server (port 8501)
```

### Docker Deployment
```
Docker Image (bank-deposit-xai:latest)
    ↓
Container Runtime
    ↓
Exposed Port 8501
    ↓
Web Browser Access
```

### Production Options
- **Streamlit Cloud**: Direct GitHub integration
- **AWS EC2**: Docker container on cloud VM
- **Heroku**: Container deployment
- **Azure App Service**: Web app hosting

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Web UI framework |
| ML Model | XGBoost | Gradient boosting classifier |
| Explainability | SHAP | Model interpretation |
| Preprocessing | Scikit-learn | LabelEncoder, metrics |
| Data Handling | Pandas | DataFrame operations |
| Visualization | Matplotlib | Plot rendering |
| Containerization | Docker | Deployment packaging |

## Key Design Decisions

### Why XGBoost?
- Superior performance on tabular data
- Handles mixed feature types well
- Fast training and inference
- Native SHAP support via TreeExplainer

### Why LabelEncoder over One-Hot?
- Reduces dimensionality (important for SHAP visualization)
- XGBoost handles ordinal encoding efficiently
- Maintains feature interpretability
- Required for SHAP TreeExplainer compatibility

### Why JSON Model Format?
- Human-readable model structure
- Cross-platform compatibility
- Required for XGBoost categorical features
- Version control friendly

### Why SHAP over LIME?
- Theoretically grounded (Shapley values)
- Consistent and locally accurate
- Optimized TreeExplainer for XGBoost
- Rich visualization library
- Global + local explanations

## Performance Considerations

### Caching Strategy
- `@st.cache_resource`: Model and encoders (loaded once)
- `@st.cache_data`: Training data (loaded once)
- SHAP explainer cached per session

### Optimization Techniques
- Sample 100 instances for global SHAP plots (vs full dataset)
- Lazy loading of SHAP visualizations
- Matplotlib backend for faster rendering
- JSON model format for quick loading

## Security Considerations

- No sensitive data in repository (.gitignore configured)
- Input validation on user-provided features
- Model artifacts regenerated from training (not committed)
- Environment variables for production secrets
- Docker non-root user execution

## Scalability

### Current Limitations
- Single-threaded Streamlit server
- In-memory model loading
- Synchronous prediction pipeline

### Future Enhancements
- FastAPI backend for async predictions
- Model serving with TensorFlow Serving or BentoML
- Redis caching for frequent predictions
- Load balancing for multiple instances
- Database for prediction logging
