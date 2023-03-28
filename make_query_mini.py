import pandas as pd
import numpy as np
import sklearn.datasets as datasets


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
def class_imbalance_check(df, label):
	label_count_list = df[label].value_counts().tolist()
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

def make_query_classificaiton(data_path, selected_model, nan_check_flag, useless_features, numeric_col, categorical_col, label_col, class_cnt, 
	class_imbalance_flag, scaler_flag):
	Query = f'Please make machine learning classification {selected_model} model using python.\nPlease generate the code in the order described below and set all SEED values (and random_state parameters) to 42.\n'

	Query += f'Data path is {data_path}. Please load this data using pandas\n' #Data load
	if nan_check_flagis not None:
		Query += 'Data have nan value. Please fill nan data to zero\n' #fill nan
	
	if useless_feautres is not None:
		Query += f'Data have useless features. Please drop {useless_features} columns.\n'#useless features drop

	Query += 'Separate 10 percent of the validation data using the train_test_split function, and set the stratify parameter of the function to True\n' #train_test split

	Query += f'Data has numeric columns such as {numeric_col}, has categoical columns such as {categorical_col}. Ignore if there is no categorical variable, else encode categorical columns using scikit-learn labelencoder\n' #labelencoding

	if scaler_flag is not None:
		Query += 'Scale data using scikit-learn MinMaxScaler.'

	Query += f'This is data information. The label to be predicted is {label_col}, and the label consists of {class_cnt} types.\n'


def make_query_regression(data_path, selected_model, nan_check_flag, useless_features, numeric_col, categorical_col, scaler_flag):
	Query = f'Please make machine learning regression {selected_model} model using python\nPlease generate the code in the order described below and set all SEED values (and random_state parameters) to 42.\n'

	Query += f'Data path is {data_path}. Please load this data using pandas\n' #Data load
	if nan_check_flagis not None:
		Query += 'Data have nan value. Please fill nan data to zero\n' #fill nan
	
	if useless_feautres is not None:
		Query += f'Data have useless features. Please drop {useless_features} columns.\n'#useless features drop

	Query += 'Separate 10 percent of the validation data using the train_test_split function, and set the stratify parameter of the function to True\n' #train_test split

	Query += f'Data has numeric columns such as {numeric_col}, has categoical columns such as {categorical_col}. Ignore if there is no categorical variable, else encode categorical columns using scikit-learn labelencoder\n' #labelencoding

	if scaler_flag is not None:
		Query += 'Scale data using scikit-learn MinMaxScaler.'



prob_type = None #'Classification' or 'Regression'
selected_model = None


iris = load_iris()
diabetes = load_diabetes()
digits = load_digits()
linnerud = load_linnerud()
wine = load_wine()
cancer = load_breast_cancer()

## data load
df = make_df(iris)

prob_type = 'Classification'
selected_model = 'RandomForest'


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


	label_col = 'target'
	nan_check_flag = nan_check(df)

	#Drop useless columns
	if useless_featues is not None:
		df.drop(useless_features, axis=1 ,inplace=True)
	df.drop(label_col, axis=1, inplace=True)
	class_cnt = class_cnt_calculate(df, label_col)
	numeric_col, categorical_col = column_type_check(df)
	class_imbalance_flag = class_imbalance_flag(df)



else: #Regression
	data_path = None
	nan_check_flag = None
	useless_features = None
	numeric_col = None
	categorical_col = None
	regression_cnt = None
	label_col = None
	scaler_flag = True
	
	label_col = 'target'
	nan_check_flag = nan_check(df)

	#Drop useless columns
	if useless_featues is not None:
		df.drop(useless_features, axis=1 ,inplace=True)
	df.drop(label_col, axis=1, inplace=True)
	class_cnt = class_cnt_calculate(df, label_col)
	numeric_col, categorical_col = column_type_check(df)



