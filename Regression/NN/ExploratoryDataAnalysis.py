import os
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# Ensure the plot directory exists
PLOT_DIR = 'Regression/NN/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Function to plot Training and Validation Loss
def plot_training_validation_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'training_validation_loss.png')
    plt.savefig(path)
    plt.close()
    print(f"Training and validation loss plot saved to {path}")

# Function to plot Actual vs Predicted Ratings
def plot_actual_vs_predicted(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.title('Actual vs. Predicted Ratings')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'actual_vs_predicted.png')
    plt.savefig(path)
    plt.close()
    print(f"Actual vs. Predicted Ratings plot saved to {path}")

# Function to plot Residuals Distribution
def plot_residuals_distribution(y_test, predictions):
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='blue')
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'residuals_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f"Residuals distribution plot saved to {path}")

# Detailed residual analysis
def plot_detailed_residuals(y_test, predictions):
    residuals = y_test - predictions

    # Scatter plot of residuals vs predicted values
    plt.figure(figsize=(12, 6))
    plt.scatter(predictions, residuals, alpha=0.3, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.tight_layout()
    path1 = os.path.join(PLOT_DIR, 'residuals_vs_predicted.png')
    plt.savefig(path1)
    plt.close()

    # Density plot of residuals
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True, color='blue', bins=30, stat='density')
    sns.kdeplot(residuals, color='red', lw=2)
    plt.title('Density Plot of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    path2 = os.path.join(PLOT_DIR, 'density_plot_residuals.png')
    plt.savefig(path2)
    plt.close()

    # Q-Q plot
    plt.figure(figsize=(12, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.tight_layout()
    path3 = os.path.join(PLOT_DIR, 'qq_plot_residuals.png')
    plt.savefig(path3)
    plt.close()

    print("Detailed residual plots saved to:")
    print(f"- {path1}")
    print(f"- {path2}")
    print(f"- {path3}")
