import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
# This function builds a simple feedforward neural network for regression tasks.
# It consists of an input layer, two hidden layers with ReLU activation, and an output layer.
# The model is compiled with Adam optimizer and mean squared error loss function.
# The input shape is determined by the number of features in the dataset.
# The model is suitable for regression tasks where the target variable is continuous.
def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# This function trains the neural network model using the training data.
# It uses mean squared error as the loss function and Adam optimizer.
# The model is trained for 100 epochs with a batch size of 32.
# The training history is returned for further analysis.
def train_model(model, X_train, y_train):
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )

    return history

