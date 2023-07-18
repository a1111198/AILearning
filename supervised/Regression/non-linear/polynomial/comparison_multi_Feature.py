import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import pandas as pd

# Download the dataset
data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')

# Define the features and the target
X = data.drop(columns='quality')
y = data['quality']

# # And then proceed with the same steps as the previous script
# ...

# # Load the Boston Housing dataset
# boston_dataset = load_boston()

# # Create DataFrame
# boston = pd.DataFrame(boston_dataset.data,
#                       columns=boston_dataset.feature_names)

# # Defining features and target variable
# X = boston
# y = boston_dataset.target

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Lists to store results
degrees = []
r2_train_values = []
r2_test_values = []

# For each degree from 1 to 5 (you can increase this range if computation is not an issue)
for degree in range(1, 3):

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Making predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Calculating R2 score
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Save results
    degrees.append(degree)
    r2_train_values.append(r2_train)
    r2_test_values.append(r2_test)

    # Print the results
    print("Degree: ", degree)
    print("Training set R^2: ", r2_train)
    print("Test set R^2: ", r2_test)

# Plot R2 score as a function of the degree of the polynomial
plt.figure(figsize=(8, 6))
plt.plot(degrees, r2_train_values, label='Training set')
plt.plot(degrees, r2_test_values, label='Test set')
plt.xlabel('Degree of Polynomial')
plt.ylabel('R^2 Score')
plt.legend()
plt.show()
