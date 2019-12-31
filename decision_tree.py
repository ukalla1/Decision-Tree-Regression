import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split																						# to break the dataset into train and test
from sklearn.tree import DecisionTreeRegressor																								# to import the decision tree class from the sklearn library
from sklearn.preprocessing import OneHotEncoder																								# To one hot encode the categorical data
from sklearn.compose import ColumnTransformer 																								# To one hot encode only one column in the data set
import matplotlib.pyplot as plt 																											# matplot library for visualization

# Read the csv into data variable
data = pd.read_csv('50_Startups.csv')

#Breaking the data into dependent(x) and independent(y) vars
x = data.iloc[:, :-1]
y = data.iloc[:, -1].values

# one hot encoding the categorical data
column_transformer = ColumnTransformer([("State", OneHotEncoder(categories='auto'), [3])], remainder = 'passthrough')
x = column_transformer.fit_transform(x)

# splitting the data into train and test data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# creating an object of class decision tree and fitting it to the training data
dt_regressor = DecisionTreeRegressor(random_state = 0)
dt_regressor.fit(x_train, y_train)

# predicting using the regressor with test data
y_pred = dt_regressor.predict(x_test)

# printing the predicitions and the actual outputs
print("{}\t\t{}\t\t\t{}". format('predicitions', 'actual', 'difference'))
for i in range(0, len(y_pred)-1):
	print("{}\t:\t\t{}\t\t{}".format(y_pred[i], y_test[i], abs(float(y_pred[i] - y_test[i]))))

# writing the decision tree graph to an png file
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(dt_regressor, out_file = dot_data, filled = True, rounded = True, special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("dtree2.png")