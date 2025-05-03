from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# This function trains a Decision Tree
def decision_tree_training_model(data):
    X = data[['year', 'num_pages', 'ratings_count', 'text_reviews_count']]
    y = data['high_rating']
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Calculate & print the accuracy of the model
    print("Accuracy score: " , accuracy_score(y_test, predictions))

    # Visualization of the Decision Tree
    feature_names = ['year', 'num_pages', 'ratings_count', 'text_reviews_count']
    class_names = ['Low Rating', 'High Rating']
    visualize_decision_tree(clf, feature_names, class_names, "./Classification/DecisionTree/plots/decision_tree.png")

# Visualize the Decision Tree and save it as an image.

def visualize_decision_tree(clf, feature_names, class_names, output_file='decision_tree.png'):
    plt.figure(figsize=(25, 20))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("Decision Tree Visualization")
    plt.savefig(output_file, dpi = 300)
    plt.close()




