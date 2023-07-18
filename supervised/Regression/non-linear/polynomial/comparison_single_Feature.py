import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Boston housing dataset
boston = load_boston()
X = boston.data[:, 5]  # 'RM' feature
y = boston.target
X = X[:, np.newaxis]  # Reshape from (506,) to (506, 1)

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

degrees = list(range(1, 51))  # List of degrees
r2_scores = []  # List to store R^2 scores

# Loop over degrees
for degree in degrees:
    # Polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_test_pred = model.predict(X_test_poly)

    # Calculate R^2 score
    r2 = r2_score(y_test, y_test_pred)
    r2_scores.append(r2)

# Plot R^2 scores
plt.figure("R^2 scores vs Polynomial degree")
plt.plot(degrees, r2_scores, marker='o')
plt.title('R^2 scores vs Polynomial degree')
plt.xlabel('Degree')
plt.ylabel('R^2 Score')
plt.grid(True)
plt.show()
