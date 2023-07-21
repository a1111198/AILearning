# Import necessary libraries
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load Boston Housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Display the features of the dataset
print("Features: ", boston.feature_names)
print("Shape of the dataset: ", X.shape)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Create a Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# Train the model using the training sets
regressor.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regressor.predict(X_test)

# Evaluating the Algorithm
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared scores:', r2_score(y_test, y_pred))
