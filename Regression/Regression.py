import time

import pandas as pd
from fontTools.mtiLib import build
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from Regression.NN.ExploratoryDataAnalysis import plot_training_validation_loss, plot_actual_vs_predicted, \
    plot_residuals_distribution, plot_detailed_residuals
from Regression.NN.NeuralNetwork import build_model, train_model


# Loading Data
def load_data(filepath):
    data = pd.read_csv(filepath, on_bad_lines = 'skip')

    #cleaning up column names
    data.columns = data.columns.str.strip()

    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
    print("First 5 rows of the dataset:")
    print(data.head())
    print(data.describe())

    return data

def preprocess_data(data):
    features = ['num_pages', 'ratings_count', 'text_reviews_count']
    target = 'average_rating'

    # Check for missing values
    print("\nCheck for missing values:")
    print(data[features + [target]].isnull().sum())
    data = data.dropna(subset=features + [target])

    # Convert 'publication_date' to datetime format
    data['publication_date_parsed'] = pd.to_datetime(data['publication_date'], errors='coerce')

    # Drop rows with invalid dates
    if data['publication_date_parsed'].isnull().any():
        invalid_dates_index = data[data['publication_date_parsed'].isnull()].index
        print(f"Rows with invalid dates: {invalid_dates_index}")
        data.drop(index=invalid_dates_index, inplace=True)

    # Extract year and drop rows with missing 'year'
    data['year'] = data['publication_date_parsed'].dt.year
    data = data.dropna(subset=['year'])

    # Split into features and target
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test





# Normalizing the data
# Using MinMaxScaler
def normalize_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled



def regression():
    filepath = 'Dataset/books.csv'
    try:
        data = load_data(filepath)
        if data is not None:
            start_time = time.time()
            X_train, X_test, y_train, y_test = preprocess_data(data)
            print("\n Preprocessing completed successfully.")
            print(f"Time taken for preprocessing: {time.time() - start_time} seconds")
            print("\n First 5 rows of data after preprocessing:")
            print(data.head())
            X_train, X_test = normalize_data(X_train, X_test)
            print("\n First 5 rows of data after normalization:")
            print(data.head())
            print("\n Data normalization completed successfully.")
            print("2. REGRESSION")
            print("-------Neural Network--------")
            model = build_model(X_train.shape[1])
            start_time = time.time()
            history = train_model(model, X_train, y_train)
            training_time = time.time() - start_time
            print(f" - Training Time: {training_time:.4f} seconds")
            predictions = model.predict(X_test).flatten()

            plot_training_validation_loss(history)
            print("\n Training Validation Loss Plot plotted successfully.")


            plot_actual_vs_predicted(y_test, predictions)
            print("\n Actual vs Predicted Plot plotted successfully.")

            plot_residuals_distribution(y_test, predictions)
            print("\n Residuals Distribution Plot plotted successfully.")

            plot_detailed_residuals(y_test, predictions)
            print("\n Detailed Residuals Plot plotted successfully.")






    except Exception as e:
        print(f"An error occured: {e}")


