import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Boston housing dataset
boston = load_boston()
X = boston.data[:, 5:6]  # "RM" feature
y = boston.target

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add a column of ones to X to represent the intercept
X = np.c_[np.ones(X.shape[0]), X]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def gradient_descent(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    history = np.zeros(epochs)

    for i in range(epochs):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - lr * gradients
        history[i] = np.sqrt(np.mean((X.dot(theta) - y)**2))  # RMSE

    return theta, history


# Run gradient descent with different learning rates
lrs = np.arange(0.1, 0.3, 0.1)
fig, ax = plt.subplots(figsize=(10, 7))

for lr in lrs:
    theta, history = gradient_descent(X_train, y_train, lr)
    ax.plot(history, label=f'Learning Rate {lr:.2f}')

ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE')
ax.legend()
plt.show()
