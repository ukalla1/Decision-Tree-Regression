This project is the implementation of decision tree regression algorithm in python using the scikit learn library.
The following are the goals of the code:
	1. To read a CSV file with the data inputs.
	2. Store the data into 2 variables based on dependent and independent vars (Here, it is assumed that the data has independent vars in all but the last column, with the data itself having 5 columns).
	3. OneHotEncode the categorical data.
	4. Split the data into train and test data (80% for train).
	5. Create a decision tree object of the decision tree class.
	6. Fit the object with the training data.
	7. Predict the output for the test data and compare it  with the actual outputs.
	8. View the tree itself.