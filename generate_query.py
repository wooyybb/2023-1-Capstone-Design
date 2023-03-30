import os
import sys
import argparse
import pandas as pd
import numpy as np
import sklearn.datasets as datasets
import optuna
from optuna import trial

 # return dataframe shape dim-0, dim-1 int data type
def return_shape(df):
	return df.shape[0], df.shape[1]

#return nan check result boolean data type
#https://suy379.tistory.com/130
def nan_check(df):
	nan_check = df.isnull().sum().tolist()

	flag = False
	for value in nan_check:
		if value !=0:
			flag = True
			break
	return flag

#https://gibles-deepmind.tistory.com/m/138 -> whether variable is numeric or categorical
def column_type_check(df):
	numeric_col = df._get_numeric_data().columns.tolist()
	categorical_col = list(set(df.columns) - set(numeric_col))

	return numeric_col, categorical_col

#class imbalance check function
def class_imbalance_check(df, label_col):
	label_count_list = df[label_col].value_counts().tolist()
	flag = False
	if max(label_count_list) > min(label_count_list) * 9:
		flag = True
	return flag
#return class_cnt
def class_cnt_calculate(df, label_col):
	class_cnt = len(df[label_col].unique())
	return class_cnt

def make_df(data):
    df = pd.DataFrame(data.data, columns = data.feature_names)
    df['target'] = data.target
    return df

def make_query_ML_classificaiton(data_path, selected_model,  label_col, class_cnt, nan_check_flag, useless_features, numeric_col, categorical_col,
	scaler_flag, class_imbalance_flag):
	Query = f'Please make machine learning classification {selected_model} model using python.\nPlease generate the code in the order described below and set all SEED values (and random_state parameters) to 42.\n'
	Query += f'First, ignore warnings using the "warinings" library.\n'
	Query += f'This is data information. The label to be predicted is {label_col}, and the label consists of {class_cnt} types.\n'

	Query += f'Data path is {data_path}. Please load this data using pandas\n' #Data load
	if nan_check_flag is not False:
		Query += 'Data have nan value. Please fill nan data to zero, But Remove columns with more than 20% missing values\n' #fill nan
	
	if useless_features is not False:
		Query += f'Data have useless features. Please drop {useless_features} columns.\n'#useless features drop

	Query += 'Separate 10 percent of the validation data using the train_test_split function, and set the stratify parameter of the function to True\n' #train_test split

	Query += f'Data has numeric columns such as {numeric_col}, has categoical columns such as {categorical_col}. Ignore if there is no categorical variable, else encode categorical columns using scikit-learn labelencoder.\n' #labelencoding

	Query += f'Encode the label data (y of train or y of validation) with labelencoder.\n'

	if scaler_flag is not False:
		Query += 'Scale data using scikit-learn MinMaxScaler.\n'

	if class_imbalance_flag is not False:
		Query += 'Since there is a current class imbalance problem, use imblearn SMOTE library to oversample only the training data.\n'

	if selected_model == "RandomForest":
		Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {param_rf} and write the objective function of optuna. Do not use pruner and use the TPESampler.\n'
	elif selected_model == "ExtraTree":
		Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {param_ex} and write the objective function of optuna. Do not use pruner and use the TPESampler.\n'
	elif selected_model == "XGBoost":
		Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {param_xgb} and write the objective function of optuna.\n'
		Query+= f'In the fit part of the internal of objective function, set eval_set to validation data and earlystopping round to 100. Do not use pruner and use the TPESampler.\n'
	elif selected_model == "LightGBM":
		Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {param_lgbm} and write the objective function of optuna. Do not use pruner and use the TPESampler.\n'

	Query += 'Set the evaluation metric inside the objective function to macro f1-score, but fit part metric is logloss.\n'

	Query += 'Last, check you have properly imported the necessary libraries.'
	return Query

def make_query_ML_regression(data_path, selected_model, label_col, regression_cnt, nan_check_flag, useless_features, numeric_col, categorical_col, scaler_flag):
	Query = f'Please make machine learning regression {selected_model} model using python\nPlease generate the code in the order described below and set all SEED values (and random_state parameters) to 42.\n'
	Query += f'First, ignore warnings using the "warinings" library.\n'
	Query += f'This is data information. The label to be predicted is {label_col}, and the label consists of {regression_cnt} types.\n'
	

	Query += f'Data path is {data_path}. Please load this data using pandas\n' #Data load
	if nan_check_flag is not None:
		Query += 'Data have nan value. Please fill nan data to zero, But Remove columns with more than 20% missing values\n' #fill nan
	
	if useless_features is not None:
		Query += f'Data have useless features. Please drop {useless_features} columns.\n'#useless features drop

	Query += 'Separate 10 percent of the validation data using the train_test_split function, and set the stratify parameter of the function to True\n' #train_test split

	Query += f'Data has numeric columns such as {numeric_col}, has categoical columns such as {categorical_col}. Ignore if there is no categorical variable, else encode categorical columns using scikit-learn labelencoder\n' #labelencoding

	if scaler_flag is not None:
		Query += 'Scale data using scikit-learn MinMaxScaler.'

	if selected_model == "RandomForest":
		Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {param_rf} and write the objective function of optuna. Do not use pruner and use the TPESampler.\n'
	elif selected_model == "ExtraTree":
		Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {param_ex} and write the objective function of optuna. Do not use pruner and use the TPESampler.\n'
	elif selected_model == "XGBoost":
		Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {param_xgb} and write the objective function of optuna. Do not use pruner and use the TPESampler.\n'
		Query+= f'In the fit part of the internal of objective function, set eval_set to validation data and earlystopping round to 100. Do not use pruner and use the TPESampler.\n'
	elif selected_model == "LightGBM":
		Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {param_lgbm} and write the objective function of optuna. Do not use pruner and use the TPESampler.\n'

	Query += 'Set the evaluation metric inside the objective function to Mean Square Error, but fit part metric is l2.\n'

	Query += 'Last, check you have properly imported the necessary libraries.'
	return Query

param_lgbm = {
	'reg_lambda': 'trial.suggest_float("reg_lambda", 1e-5, 1.0)',
	'reg_alpha': 'trial.suggest_float("reg_alpha", 1e-5, 1.0)',
	'max_depth': 'trial.suggest_int("max_depth", 4,100)',
	'colsample_bytree': 'trial.suggest_float("colsample_bytree", 0.1, 1)',
	'subsample': 'trial.suggest_loguniform("subsample", 0.5, 1)',
	'learning_rate': 'trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)',
	'n_estimators': 'trial.suggest_int("n_estimators", 100, 3000)',
	'min_child_samples': 'trial.suggest_int("min_child_samples", 5, 100)',
	'random_state' : 42
	}
param_xgb = {
        'reg_lambda': 'trial.suggest_float("reg_lambda", 1e-5, 1.0)',
        'reg_alpha': 'trial.suggest_float("reg_alpha", 1e-5, 1.0)',
        'max_depth': 'trial.suggest_int("max_depth", 4,100)',
        'colsample_bytree': 'trial.suggest_float("colsample_bytree", 0.1, 1)',
        'subsample': 'trial.suggest_float("subsample", 0.5, 1)',
        'learning_rate': 'trial.suggest_float("learning_rate",1e-5, 1e-1)',
        'n_estimators': 'trial.suggest_int("n_estimators", 100, 3000)',
        'min_child_weight': 'trial.suggest_int("min_child_weight", 1, 50)',
        'gpu_id': -1,  # CPU 사용시
	'random_state' : 42
	}
param_ex = {
        'max_depth': 'trial.suggest_int("max_depth", 10, 100)',
        'min_samples_split': 'trial.suggest_int("min_samples_split", 2, 150)',
        #'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 1000),
        'n_estimators': 'trial.suggest_int("n_estimators", 100, 2500)',
        'criterion' : 'trial.suggest_categorical("criterion", ["gini","entropy"])',
        'bootstrap' : True,
        'random_state': 42,
   	}
param_rf = {
        'max_depth': 'trial.suggest_int("max_depth", 10, 100)',
        'min_samples_split': 'trial.suggest_int("min_samples_split", 2, 150)',
        #'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 1000),
        'n_estimators': 'trial.suggest_int("n_estimators", 100, 2500)',
        'criterion' : 'trial.suggest_categorical("criterion", ["gini","entropy"])',
        'random_state': 42,
   	}




prob_type = None #'Classification' or 'Regression'
selected_model = None


'''iris = load_iris()
diabetes = load_diabetes()
digits = load_digits()
linnerud = load_linnerud()
wine = load_wine()
cancer = load_breast_cancer()

## data load
df = make_df(iris)'''

if __name__ == '__main__':

	prob_type = 'Classification'
	selected_model = ['RandomForest','ExtraTree','XGBoost','LightGBM']

	if prob_type == 'Classification':
		data_path = None
		nan_check_flag = None
		useless_features = None
		numeric_col = None
		categorical_col = None
		label_col = None
		class_cnt = None
		class_imbalance_flag = None
		scaler_flag = True

		#pca_flag = None

		data_path = './iris.csv'
		label_col = 'Species'
		useless_features = ['Id']

		df = pd.read_csv(data_path)
		nan_check_flag = nan_check(df)
		#Drop useless columns
		if useless_features is not None:
			df.drop(useless_features, axis=1 ,inplace=True)
		class_cnt = class_cnt_calculate(df, label_col)
		class_imbalance_flag = class_imbalance_check(df, label_col)
		df.drop(label_col, axis=1, inplace=True)
		numeric_col, categorical_col = column_type_check(df)

		for model in selected_model:
			if model == 'LightGBM':
				param_lgbm['num_class'] = class_cnt
			elif model == 'XGBoost':
				param_xgb['num_class'] = class_cnt
			Query = make_query_ML_classificaiton(data_path, model, label_col, class_cnt, nan_check_flag, useless_features, numeric_col, categorical_col, scaler_flag, class_imbalance_flag)

			if model == 'LightGBM':
				del param_lgbm['num_class']
			elif model == 'XGBoost':
				del param_xgb['num_class']
			print(Query)
	else: #Regression
		data_path = None
		nan_check_flag = None
		useless_features = None
		numeric_col = None
		categorical_col = None
		label_col = None
		scaler_flag = True
		
		label_col = 'target' #Consider multivariate regression
		nan_check_flag = nan_check(df)

		#Drop useless columns
		if useless_features is not None:
			df.drop(useless_features, axis=1 ,inplace=True)
		df.drop(label_col, axis=1, inplace=True)
		numeric_col, categorical_col = column_type_check(df)



