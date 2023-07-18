import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Load Boston housing dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target

print("\n----- Dataset Loaded -----\n")
print(data.head())

# Feature and target
X = data[['RM']]
Y = data['MEDV']

print("\n----- Features and Target Defined -----\n")
print("Features:\n", X.head())
print("Target:\n", Y.head())

# Split the data into training/testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=5)

print("\n----- Data Split into Training and Testing Sets -----\n")
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))

# Create linear regression object
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, Y_train)

print("\n----- Model Trained -----\n")
print("Intercept (β0):", model.intercept_)
print("Coefficient (β1):", model.coef_[0])

# Make predictions using the testing set
Y_pred = model.predict(X_test)

print("\n----- Predictions Made on Testing Set -----\n")
print("First 5 predictions:", Y_pred[:5])

# Plot outputs
plt.figure("House Price Predictions vs Actual")
plt.scatter(X_test, Y_test, color='black', label="Actual Prices")
plt.plot(X_test, Y_pred, color='blue', linewidth=3, label="Predicted Prices")
plt.xlabel('Number of Rooms')
plt.ylabel('Median Value')
plt.title('Rooms vs Value')
plt.legend()
plt.show()

# Calculate residuals
residuals = Y_test - Y_pred

# Plot histogram of residuals
plt.figure("Histogram of Residuals")
plt.hist(residuals, bins=20)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()
