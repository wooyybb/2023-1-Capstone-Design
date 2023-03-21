import pandas as pd
import numpy as np
import sklearn.datasets as datasets

def return_shape(df): # return dataframe shape dim-0, dim-1 int data type
	return df.shape[0], df.shape[1]

def check_nan(df): #return nan check result boolean data type
	nan_check = df.isnull().sum().tolist()

	flag = False
	for value in nan_check:
		if value !=0:
			flag = True
			break
	return flag

#https://gibles-deepmind.tistory.com/m/138 -> whether variable is numeric or categorical
def column_type_check(df)
	numeric_col = df._get_numeric_data().columns.tolist()
	categorical_col = list(set(df.columns) - set(numeric_col))

	return numeric_col, categorical_col



iris = load_iris()
diabetes = load_diabetes()
digits = load_digits()
linnerud = load_linnerud()
wine = load_wine()
cancer = load_breast_cancer()

Query = "Please make neural network code using pytorch library."



