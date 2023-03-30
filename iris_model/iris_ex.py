import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import optuna

# Load data
data = pd.read_csv('./iris.csv')

# Drop the 'Id' column
data.drop('Id', axis=1, inplace=True)

# Separate the target variable
target = data['Species']
data.drop('Species', axis=1, inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
if len(data.select_dtypes(include=['object']).columns) > 0:
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = encoder.fit_transform(data[col])

# Scale the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.1, stratify=target, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2500),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'bootstrap': True,
        'random_state': 42
    }
    
    model = ExtraTreesClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    score = f1_score(y_valid, y_pred, average='macro')
    return score

# Optimize hyperparameters with Optuna
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

# Train the model with the optimal hyperparameters
opt_params = study.best_params
model = ExtraTreesClassifier(**opt_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
score = f1_score(y_valid, y_pred, average='macro')

print(f'The F1 score of the final model is {score:.3f}.')