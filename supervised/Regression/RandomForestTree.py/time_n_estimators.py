import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import time
from joblib import parallel_backend

# Load the MNIST dataset from a CSV file
data = pd.read_csv(r'C:\Projects\AIlearnings\public_data\mnist_784.csv')
# Assuming the CSV file has a 'label' column for the target
y = data['class']
# Drop the label column to obtain only features
X = data.drop('class', axis=1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Prints the first few rows of the dataset
print(X_train.head())

# Lists to store cross-validated scores and time durations
cv_scores = []
train_times = []
predict_times = []

# Range of 'n_estimators' values to explore
n_estimators = [10, 50, 100, 200, 500]

for n in n_estimators:
    # Initialize the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=n)

    # Record the start time
    start = time.time()

    # Using GPU backend for computation, if available
    with parallel_backend('threading', n_jobs=-1):
        # Train the model
        rf.fit(X_train, y_train)

    # Record the end time and calculate the duration
    end = time.time()
    duration = end - start

    # Append the training time to its respective list
    train_times.append(duration)

    # Record the start time for prediction
    start = time.time()

    # Predict the class of the test set
    y_pred = rf.predict(X_test)

    # Record the end time and calculate the duration
    end = time.time()
    duration = end - start

    # Append the prediction time to its respective list
    predict_times.append(duration)

    # Using GPU backend for computation, if available
    with parallel_backend('threading', n_jobs=-1):
        cv_score = cross_val_score(rf, X, y, cv=3).mean()

    # Append the cross-validated score to its respective list
    cv_scores.append(cv_score)

# Plotting accuracy scores
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(n_estimators, cv_scores)
plt.xlabel('Number of Estimators')
plt.ylabel('Cross-Validated Accuracy')

# Plotting training time performance
plt.subplot(132)
plt.plot(n_estimators, train_times)
plt.xlabel('Number of Estimators')
plt.ylabel('Training Time (s)')

# Plotting prediction time performance
plt.subplot(133)
plt.plot(n_estimators, predict_times)
plt.xlabel('Number of Estimators')
plt.ylabel('Prediction Time (s)')

plt.tight_layout()
plt.show()
