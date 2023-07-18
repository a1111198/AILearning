from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
# Load the Boston Housing dataset
X, y = load_boston(return_X_y=True)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Train a Lasso model
lasso_model = Lasso(alpha=0.1)  # alpha is equivalent to lambda in the formula
lasso_model.fit(X_train, y_train)

# Train a Ridge model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Print the training and test MSE for each model
for model in [linear_model, lasso_model, ridge_model]:
    print(
        f"Training MSE for {model.__class__.__name__}: {mean_squared_error(y_train, model.predict(X_train))}")
    print(
        f"Test MSE for {model.__class__.__name__}: {mean_squared_error(y_test, model.predict(X_test))}\n")
