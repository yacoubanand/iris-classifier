import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import os

def train():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test):.2f}")

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs('outputs', exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')

    joblib.dump(model, 'outputs/model.joblib')
    print("Model and confusion matrix saved to outputs/")

if __name__ == "__main__":
    train()