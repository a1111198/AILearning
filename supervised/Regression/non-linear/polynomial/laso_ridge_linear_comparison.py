import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

# Load and split the data
boston = load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Prepare Polynomial Features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize models
linear_model = LinearRegression()
lasso_model = Lasso(alpha=0.1)
ridge_model = Ridge(alpha=0.1)
poly_model = LinearRegression()

# Train models
linear_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
poly_model.fit(X_train_poly, y_train)

# Make predictions
linear_preds = linear_model.predict(X_test)
lasso_preds = lasso_model.predict(X_test)
ridge_preds = ridge_model.predict(X_test)
poly_preds = poly_model.predict(X_test_poly)

# Calculate R-squared values
linear_r2 = r2_score(y_test, linear_preds)
lasso_r2 = r2_score(y_test, lasso_preds)
ridge_r2 = r2_score(y_test, ridge_preds)
poly_r2 = r2_score(y_test, poly_preds)

print(f"Linear R-squared: {linear_r2}")
print(f"Lasso R-squared: {lasso_r2}")
print(f"Ridge R-squared: {ridge_r2}")
print(f"Polynomial R-squared: {poly_r2}")
