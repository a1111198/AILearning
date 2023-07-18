import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Boston housing dataset
boston = load_boston()
X = boston.data[:, 5]  # 'RM' feature
y = boston.target
X = X[:, np.newaxis]  # Reshape from (506,) to (506, 1)

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create a PolynomialFeatures object with degree 2
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit a linear regression model to the transformed data
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Print the model coefficients
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# Make predictions
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calculate mean squared error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f'Training MSE: {mse_train:.2f}')
print(f'Test MSE: {mse_test:.2f}')

# Plotting the polynomial regression model
plt.figure("Polynomial Regression")
plt.scatter(X_train, y_train, color='blue', label='Actual')
plt.scatter(X_train, y_train_pred, color='red', label='Predicted')
plt.title('Polynomial Regression with degree=2')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.legend()
plt.show()
