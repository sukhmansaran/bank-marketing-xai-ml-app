import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

# Encode categorical variables
print("Encoding categorical variables...")
non_numeric_columns = X_train.select_dtypes(include=['object']).columns
le_dict = {}

for column in non_numeric_columns:
    le = LabelEncoder()
    le.fit(X_train[column])
    X_train[column] = le.transform(X_train[column])
    X_test[column] = le.transform(X_test[column])
    le_dict[column] = le

# Scale numeric features
print("Scaling numeric features...")
numeric_columns = X_train.select_dtypes(exclude=['object']).columns
scaler = StandardScaler()
scaler.fit(X_train[numeric_columns])

X_train[numeric_columns] = scaler.transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Train model
print("Training model...")
# using random forest classifier for model training hypertuning included
rf = RandomForestClassifier()

param_grid = {'n_estimators': [100, 200, 300, 400],
                 'max_depth': [5, 10, 15, 20]}

grid_search = GridSearchCV(rf, param_grid = param_grid, scoring = 'f1', cv = 5, n_jobs = -1)
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
print(f"AUC-ROC:  {roc:.4f}")

# Save model and preprocessors
print("\nSaving model and preprocessors...")
with open('models/model_rf.pkl', 'wb') as f:
    pickle.dump(best_model, f)

preprocessors = {
    'label_encoders': le_dict,
    'scaler': scaler,
    'numeric_columns': numeric_columns,
    'non_numeric_columns': non_numeric_columns
}

with open('models/preprocessors_rf.pkl', 'wb') as f:
    pickle.dump(preprocessors, f)

print("✅ Model training complete!")
