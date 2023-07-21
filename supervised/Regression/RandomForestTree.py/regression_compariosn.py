# Import required libraries
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

# Load the California Housing dataset
data = fetch_california_housing()
X = data['data']
y = data['target']
print("DATA FETCHED")
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Range of 'n_estimators' values to explore
n_estimators = [10, 50, 100, 200, 300]

# Range of degrees for Polynomial Regression
# using only degrees 1 and 2 for large datasets due to computation complexity
degrees = [1, 2, 3, 4]

# For each value of 'n_estimators', train a RandomForestRegressor and record performance
for n in n_estimators:
    # Initialize and train the Random Forest regressor
    rf = RandomForestRegressor(n_estimators=n)

    # Record the start time
    start_train = time.time()

    # Train the model
    rf.fit(X_train, y_train)

    # Record the end time and calculate the duration
    end_train = time.time()
    train_time_rf = end_train - start_train

    # Record the start time for prediction
    start_pred = time.time()

    # Make predictions using the testing set
    y_pred_rf = rf.predict(X_test)

    # Record the end time and calculate the duration
    end_pred = time.time()
    pred_time_rf = end_pred - start_pred

    # Print R-squared scores and time performance
    print(
        f"Random Forest with {n} estimators: R2 score: {r2_score(y_test, y_pred_rf)}, Training time: {train_time_rf}, Prediction time: {pred_time_rf}")


# For each degree, train a Polynomial Regression model and record performance
for d in degrees:
    # Generate Polynomial Features
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Initialize the Linear Regression model
    lr = LinearRegression()

    # Record the start time
    start_train = time.time()

    # Train the model
    lr.fit(X_train_poly, y_train)

    # Record the end time and calculate the duration
    end_train = time.time()
    train_time_lr = end_train - start_train

    # Record the start time for prediction
    start_pred = time.time()

    # Make predictions using the testing set
    y_pred_lr = lr.predict(X_test_poly)

    # Record the end time and calculate the duration
    end_pred = time.time()
    pred_time_lr = end_pred - start_pred

    # Print R-squared scores and time performance
    print(
        f"Polynomial Regression of degree {d}: R2 score: {r2_score(y_test, y_pred_lr)}, Training time: {train_time_lr}, Prediction time: {pred_time_lr}")
