import pandas as pd
from seaborn import pairplot
from sklearn.preprocessing import MinMaxScaler
import time

from Classification.DecisionTree.DecisionTree import decision_tree_training_model
from Classification.DecisionTree.ExploratoryDataAnalysis import correlation_matrix, pair_plot, box_plot, plot_density, \
    plot_time_series, scatter_plot_matrix
from Classification.SVM.ExploratoryDataAnalysis import correlation_matrix_plot, feature_distribution_plot, \
    model_performance_plot, confustion_matrix_plot
from Classification.SVM.SVM import svm_training_model


# This function loads and explores the data
# It creates a new column titled 'high_rating'
# This column will be used to classify the books based on their average rating
# The threshold for high rating is set to 4.0
def load_and_explore_data(filepath):
    data = pd.read_csv(filepath, sep=",", on_bad_lines='skip')
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
    print("First 5 rows of the dataset:")
    print(data.head())
    print(data.describe())

    if data['publication_date'].isnull().any():
        print("Rows with invalid dates: ")
        print(data[data['publication_date'].isnull()])

    # Creating a new column titled 'high_rating'
    # This column will be used to classify the books based on their average rating
    # The threshold for high rating is set to 4.0

    data['high_rating'] = (data['average_rating'] >= 4).astype(int)
    return data


# This function preprocesses the data
# It drops irrelevant columns and handles missing values
def preprocess_data(data, relevant_columns):
    print("\n Check for missing values: ")
    print(data[relevant_columns].isnull().sum())
    data = data.dropna(subset=relevant_columns)
    ## Convert 'publication_date' to datetime format
    #coerce = invalid parsing =>  set as NaN
    data['publication_date_parsed'] = pd.to_datetime(data['publication_date'], errors='coerce')

    # Check for invalid dates
    # If there are invalid dates, drop those rows
    if data['publication_date_parsed'].isnull().any():
        invalid_dates_index = data[data['publication_date_parsed'].isnull()].index
        print(f"Rows with invalid dates: {invalid_dates_index}")
        data.drop(index = invalid_dates_index, inplace=True)

    # Extract year from publication_date
    data['year'] = data['publication_date_parsed'].dt.year
    data = data.dropna(subset = relevant_columns + ['year'])
    return data


# This function normalizes the numeric columns using MinMaxScaler
# It scales the values to a range between 0 and 1
def normalize_data(data, numeric_columns):
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data



def classification():
    print("\n Book Data Analysis - Decision Tree Classifier")
    print("--------------------------------------------------")
    numeric_columns = ['num_pages', 'ratings_count', 'text_reviews_count']
    relevant_columns = ['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count', 'publication_date']
    try:
        data = load_and_explore_data('Dataset/books.csv')
        data = preprocess_data(data, relevant_columns)
        data = normalize_data(data, numeric_columns)
        start_time = time.time()
        print("\n Preprocessing completed successfully.")
        print(f"Time taken for preprocessing: {time.time() - start_time} seconds")
        print("\n First 5 rows of data after preprocessing:")
        print(data.head())

        print("1. CLASSIFICATION")
        print("-------Decision Tree--------")

        # Using Exploratory Data Analysis (EDA) to visualize the data
        # to understand feature distributions, relationships, and correlations.
        # This helps in selecting the most relevant features to train the model.
        correlation_matrix(data)
        print("\n Correlation matrix plotted successfully.")

        pair_plot(data)
        print("\n Pair plot plotted successfully.")

        box_plot(data)
        print("\n Box plot plotted successfully.")

        plot_density(data)
        print("\n Kernel density plot plotted successfully.")

        plot_time_series(data)
        print("\n Time series plot plotted successfully.")

        scatter_plot_matrix(data)
        print("\n Scatter plot matrix plotted successfully.")

        #Now that we have explored the data and visualized the relationships between different attributes,
        # we can proceed to train the Decision Tree Classifier.
        # Call the decision tree training model function
        decision_tree_training_model(data)
        training_time_decision_tree = time.time() - start_time
        print("\n Decision Tree Classifier trained successfully.")

        print(f"Decision Tree - Training Time: {training_time_decision_tree:.4f} seconds")

        print("-------SVM--------")

        # Using Exploratory Data Analysis (EDA) to visualize the data
        # to understand feature distributions, relationships, and correlations.
        # This helps in selecting the most relevant features to train the model.
        feature_distribution_plot(data)
        print("\n Feature distribution plot plotted successfully.")

        correlation_matrix_plot(data)
        print("\n Correlation matrix plotted successfully.")

        #Now that we have explored the data and visualized the relationships between different attributes,
        # we can proceed to train the Decision Tree Classifier.
        # Call the decision tree training model function
        start_time_svm = time.time()
        clf, X_test, y_test = svm_training_model(data)
        training_time_svm = time.time() - start_time_svm
        print(f"SVM - Training Time: {training_time_svm:.4f} seconds")

        print("\n SVM Classifier trained successfully.")

        # After training the SVM classifier, we visualize its performance
        # First, we plot the overall accuracy
        model_performance_plot(clf, X_test, y_test)
        print("\n Model performance plot plotted successfully.")

        # Then, we plot the confusion matrix to analyze classification results in more detail
        confustion_matrix_plot(clf, X_test, y_test)
        print("\n Confusion matrix plot plotted successfully.")







    except Exception as e:
        print(f"Error loading data: {e}")


