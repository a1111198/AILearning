from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# Load dataset
boston_dataset = load_boston()

# Create DataFrame
boston = pd.DataFrame(boston_dataset.data,
                      columns=boston_dataset.feature_names)

# Defining features and target variable
X = boston[['RM', 'LSTAT']]
y = boston_dataset.target

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Print the coefficients
coefficients = model.coef_
print("Coefficients: ", coefficients)

# Making predictions
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calculating MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print("Training set MSE: ", mse_train)
print("Test set MSE: ", mse_test)

# Calculating R2 score
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print("Training set R^2 score: ", r2_train)
print("Test set R^2 score: ", r2_test)
