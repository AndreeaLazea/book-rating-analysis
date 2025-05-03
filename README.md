# 📚 Analyzing Book Features and Ratings Using Decision Tree, SVM, and Neural Networks

This project explores the relationship between various book features (such as number of pages, review counts, etc.) and their average ratings using multiple machine learning techniques including **Decision Trees**, **Support Vector Machines (SVMs)**, and **Neural Networks (NNs)**. The goal is to both classify and predict book ratings based on metadata.

---

## 📁 Project Structure

.
├── Classification
│   ├── DecisionTree
│   │   ├── plots/                        # Visualizations for Decision Tree classification
│   │   ├── DecisionTree.py              # Core logic for Decision Tree classification
│   │   └── ExploratoryDataAnalysis.py   # EDA specific to Decision Tree
│   ├── SVM
│   │   ├── plots/                        # Visualizations for SVM classification
│   │   ├── SVM.py                        # Core logic for SVM classification
│   │   └── ExploratoryDataAnalysis.py   # EDA specific to SVM
│   └── Classification.py                # Entry point or orchestration script for classification
├── Dataset
│   └── books.csv                         # Raw dataset used for analysis
├── Documentation/                        # Additional project documentation (if any)
├── Presentation/                         # Final presentation slides or files
├── Regression
│   └── NN
│       ├── plots/                        # Visualizations for Neural Network regression
│       ├── ExploratoryDataAnalysis.py   # EDA for regression
│       ├── NeuralNetwork.py             # NN model definition and training
│       ├── Regression.py                # Complete regression pipeline
├── main.py                               # Entry point to run the full analysis
├── README.md                             # Project overview and instructions
└── .venv/                                 # Virtual environment


---

## 📌 Objectives

- Analyze book metadata to identify factors influencing average ratings.
- Use classification algorithms (Decision Tree, SVM) to predict rating categories.
- Use regression with neural networks to predict exact average rating values.
- Visualize relationships and residuals for interpretability and insight.

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AndreeaLazea/book-rating-analysis.git
cd book-rating-analysis
