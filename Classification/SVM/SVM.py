from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC




from sklearn.svm import SVC

def svm_training_model(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    return clf


