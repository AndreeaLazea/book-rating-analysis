from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC


#Histplot
def feature_distribution_plot(data):
    features = ['num_pages', 'ratings_count', 'text_reviews_count']
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], bins=30, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

def correlation_matrix_plot(data):
    plt.figure(figsize=(10,8))
    correlation = data[['num_pages', 'ratings_count', 'text_reviews_count', 'high_rating']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Model Performance Plot
def model_performance_plot(model, X_test, y_test):
    accuracy = accuracy_score(y_test, model.predict(X_test))
    plt.figure(figsize=(8, 5))
    sns.barplot(x=['SVM Model Accuracy'], y=[accuracy])
    plt.title('Model Performance')
    plt.show()

#Confusion matrix Plot
def confustion_matrix_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()



