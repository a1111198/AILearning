# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(f"Features: {iris.feature_names}")
print(f"Target classes: {iris.target_names}")
print(f"Shape of feature data: {X.shape}")
print(f"Shape of target data: {y.shape}")

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)  # 80% training and 20% test

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

# Predicting the test set results
y_pred = log_reg.predict(X_test)

# Checking the performance of the model
print(classification_report(y_test, y_pred))
