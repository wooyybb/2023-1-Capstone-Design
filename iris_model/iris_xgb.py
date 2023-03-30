import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Load data and drop 'Id' column
data = pd.read_csv('./iris.csv')
data.drop('Id', axis=1, inplace=True)

# Separate target label from features
X = data.drop('Species', axis=1)
y = data['Species']

# Encode label data with labelencoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Scale data using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define objective function for Optuna
def objective(trial):
    # Define parameters to optimize
    params = {
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 4, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gpu_id': -1,
        'random_state': 42,
        'num_class': 3
    }

    # Create XGBClassifier with the optimized parameters
    model = XGBClassifier(**params)

    # Train the model on the training data with early stopping on the validation data
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

    # Make predictions on the validation data and calculate macro f1-score
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro')

    return score

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the best score
print('Best Score:', study.best_value)
print('Best Parameters:', study.best_params)

# Create XGBClassifier with the best hyperparameters and train on the full training data
best_params = study.best_params
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Evaluate the model on the validation data
y_pred = model.predict(X_val)
score = f1_score(y_val, y_pred, average='macro')
print('Final Score:', score)
