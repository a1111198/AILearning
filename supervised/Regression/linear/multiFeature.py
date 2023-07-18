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

# Features and target
X = data[['RM', 'LSTAT', 'PTRATIO']]
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
print("Coefficients (β1, β2, β3):", model.coef_)

# Make predictions using the testing set
Y_pred = model.predict(X_test)

print("\n----- Predictions Made on Testing Set -----\n")
print("First 5 predictions:", Y_pred[:5])

# Calculate residuals
residuals = Y_test - Y_pred

# Plot histogram of residuals
plt.figure("Histogram of Residuals")
plt.hist(residuals, bins=20, alpha=0.5, color='g', edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True)
plt.show()
