from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()

# To turn it into a binary problem, we will only classify versicolor or not, so labels will be 0 and 1
X = iris["data"][(iris["target"] == 1) | (
    iris["target"] == 0)]  # petal length, petal width
y = iris["target"][(iris["target"] == 1) | (iris["target"] == 0)]

# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Apply feature scaling to the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Logistic Regression classifier
clf = LogisticRegression(random_state=0)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Print the accuracy
print("Accuracy: ", accuracy_score(y_test, y_pred))
