import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# Load data
print("Loading data...")
train_df = pd.read_csv('data/processed/train_df.csv')
test_df = pd.read_csv('data/processed/test_df.csv')

# Separate features and target
X_train = train_df.drop('y', axis=1)
y_train = train_df['y']

X_test = test_df.drop('y', axis=1)
y_test = test_df['y']

# Convert target to numeric (yes=1, no=0)
y_train = (y_train == 'yes').astype(int)
y_test = (y_test == 'yes').astype(int)

# Encode categorical variables using LabelEncoder (required for SHAP compatibility)
print("Encoding categorical variables...")
non_numeric_columns = X_train.select_dtypes(include=['object']).columns
label_encoders = {}

for col in non_numeric_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Train model
print("Training model...")
# Using XGBoost without native categorical support (for SHAP compatibility)
xgb = XGBClassifier(enable_categorical=False, random_state=42)

param_grid = {
    'max_depth': [4, 8, 12, 16, 20],
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.1, 0.05, 0.01],
    'min_child_weight':[1, 5, 10, 15, 20]
    }

grid_search = GridSearchCV(xgb, param_grid = param_grid, scoring = 'f1', cv = 5, n_jobs = -1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {roc:.4f}")

# Save model as JSON
print("\nSaving model and preprocessors...")
best_model.save_model('models/model_xgb.json')

# Save label encoders
with open('models/label_encoders_xgb.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Model training complete!")
print(f"Model saved to: models/model_xgb.json")
print(f"Label encoders saved to: models/label_encoders_xgb.pkl")

