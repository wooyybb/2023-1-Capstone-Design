# import libraries and ignore warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score
import lightgbm as lgb
import optuna

# load data and drop 'Id' column
data_path = './iris.csv'
df = pd.read_csv(data_path)
df.drop(['Id'], axis=1, inplace=True)

# split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(df.drop(['Species'], axis=1), df['Species'], test_size=0.1, stratify=df['Species'], random_state=42)

# encode categorical variables (if any)
categorical_cols = [] # no categorical variables in this dataset
if categorical_cols:
    le = LabelEncoder()
    for col in categorical_cols:
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col] = le.transform(X_val[col])

# encode labels with label encoder
le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train)
y_val = le_y.transform(y_val)

# scale data with MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# define objective function for optuna to optimize hyperparameters
def objective(trial):
    params = {
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 4, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'subsample': trial.suggest_loguniform('subsample', 0.5, 1),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42,
        'num_class': 3
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False, eval_metric='logloss')
    y_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    
    return macro_f1

# optimize hyperparameters with optuna
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

# train and evaluate the model with the optimized hyperparameters
best_params = study.best_params
best_params['random_state'] = 42
best_params['num_class'] = 3
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False, eval_metric='logloss')
y_pred = model.predict(X_val)
macro_f1 = f1_score(y_val, y_pred, average='macro')
print('Macro F1-Score:', macro_f1)
