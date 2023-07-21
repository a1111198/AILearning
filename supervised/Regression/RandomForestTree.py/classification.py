# Import necessary libraries
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree

# Load iris dataset
iris = load_iris()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3)

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

# Prediction on test set
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Print the Classification Report
print(classification_report(y_test, y_pred))

# Print the Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Let's print one of the trees in the forest
tree.plot_tree(clf.estimators_[0], filled=True)
plt.show()

# We can also find feature importance
feature_imp = pd.Series(clf.feature_importances_,
                        index=iris.feature_names).sort_values(ascending=False)
print(feature_imp)
