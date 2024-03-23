import pandas as pd
import numpy as np
import pickle
import sklearn.ensemble as ske
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
import joblib
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from twilio.rest import Client
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
import os

# Function to load data from a CSV file
def load_data_from_csv(file_path):
    try:
        data = pd.read_csv(file_path, sep='|')
        return data
    except Exception as e:
        print(f'Error loading data: {e}')
        return None

# Function to train models and evaluate performance
def train_models_and_evaluate(X_train, y_train, X_test, y_test):
    algorithms = {
        "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
        "RandomForest": ske.RandomForestClassifier(n_estimators=50),
        "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
        "AdaBoost": ske.AdaBoostClassifier(n_estimators=100)
    }

    results = {}
    precision_scores = {}  # Store precision scores for each algorithm
    malicious_counts = {}  # Store counts of malicious files detected by each algorithm

    for algo in algorithms:
        clf = algorithms[algo]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        precision = precision_score(y_test, y_pred)  # Calculate precision
        results[algo] = score
        precision_scores[algo] = precision
        # Calculate the count of malicious files detected
        malicious_counts[algo] = np.sum((y_pred == 1) & (y_test == 1))

    # Custom algorithm for malware detection
    custom_algo = LogisticRegression(max_iter=1000)

    # Train the custom algorithm
    custom_algo.fit(X_train, y_train)

    # Predict using the custom algorithm
    y_pred_custom = custom_algo.predict(X_test)

    # Calculate accuracy and precision for the custom algorithm
    accuracy_custom = custom_algo.score(X_test, y_test)
    precision_custom = precision_score(y_test, y_pred_custom)

    # Calculate the count of malicious files detected by the custom algorithm
    malicious_count_custom = np.sum((y_pred_custom == 1) & (y_test == 1))

    # Add the custom algorithm to the dictionary of algorithms
    algorithms["logisticregression"] = custom_algo
    results["logisticregression"] = accuracy_custom
    precision_scores["logisticregression"] = precision_custom
    malicious_counts["logisticregression"] = malicious_count_custom

    winner = max(results, key=results.get)
    print('Winner algorithm is %s with a %f %% success' %
          (winner, results[winner] * 100))

    # Print accuracy and precision for each algorithm
    for algo in results:
        print(f'{algo}: Accuracy = {results[algo] * 100:.2f}%, Precision = {precision_scores[algo] * 100:.2f}%')

    # Print counts of malicious files detected by each algorithm
    print("\nNumber of Malicious Files Detected:")
    for algo, count in malicious_counts.items():
        print(f'{algo}: {count} out of {np.sum(y_test == 1)} malicious files')

    # Plot accuracy and precision
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(results.keys(), results.values())
    plt.title('Accuracy of Algorithms')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.subplot(1, 2, 2)
    plt.bar(precision_scores.keys(), precision_scores.values())
    plt.title('Precision of Algorithms')
    plt.ylabel('Precision')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Save the algorithm and the feature list for later predictions
    print('Saving algorithm and feature list in classifier directory...')
    joblib.dump(algorithms[winner], 'classifier/classifier1.pkl')
    open('classifier/features1.pkl', 'wb').write(pickle.dumps(features))
    print('Saved...')

    clf = algorithms[winner]
    res = clf.predict(X_test)
    mt = confusion_matrix(y_test, res)

    print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
    print('False negative rate : %f %%' % ((mt[1][0] / float(sum(mt[1])) * 100)))

    # Plot confusion matrix in blue color
    fig, ax = plot_confusion_matrix(conf_mat=mt, figsize=(6, 6), cmap=plt.cm.Blues)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    # Output selected features
    print('\nSelected Features:')


