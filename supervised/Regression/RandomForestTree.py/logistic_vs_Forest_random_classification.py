import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

# Initialize the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
}

# Store training times, prediction times, and scores
training_times = []
prediction_times = []
scores = []

for classifier_name, classifier in classifiers.items():
    # Record the start time for training
    start = time.time()

    # Using GPU backend for computation, if available
    with parallel_backend('threading', n_jobs=-1):
        # Train the model
        classifier.fit(X_train, y_train)

    # Record the end time and calculate the training duration
    end = time.time()
    training_time = end - start

    # Record the start time for prediction
    start = time.time()

    # Predict the class of the test set
    y_pred = classifier.predict(X_test)

    # Record the end time and calculate the prediction duration
    end = time.time()
    prediction_time = end - start

    # Score of the model
    score = classifier.score(X_test, y_test)

    # Append results to respective lists
    training_times.append(training_time)
    prediction_times.append(prediction_time)
    scores.append(score)

    print(f"{classifier_name}\nTraining time: {training_time}s\nPrediction time: {prediction_time}s\nScore: {score}")
