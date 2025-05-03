from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC




# Training the SVM model
def svm_training_model(data):
    X = data[['year', 'num_pages', 'ratings_count', 'text_reviews_count']]
    y = data['high_rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a SVM classifier
    clf = SVC(kernel='linear', random_state=42)

    # Fit the model to the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    prediction = clf.predict(X_test)


    # Calculate and print the accuracy of the model
    accuracy = clf.score(X_test, y_test)
    print("Accuracy score: ", accuracy)

    return clf, X_test, y_test


