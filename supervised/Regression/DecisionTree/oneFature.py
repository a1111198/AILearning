# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the breast cancer dataset
data = datasets.load_breast_cancer()

# Print some information about the dataset
print("Feature names:")
print(data.feature_names)
print("\nTarget names:")
print(data.target_names)

# The input data/features
X = data.data
# The target variable
y = data.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create a DecisionTreeClassifier object
clf = DecisionTreeClassifier(criterion="entropy")

# Train DecisionTreeClassifier
clf = clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Print the classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Print the Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
