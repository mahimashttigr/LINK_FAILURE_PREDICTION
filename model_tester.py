from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = "models"
MODEL_FILES = ["RandomForest.pkl", "LogisticRegression.pkl", "RBFSVM.pkl", "QDA.pkl"]

def plot_decision_boundary(model, X, Y, model_name):
    plt.figure(figsize=(8,6))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    # Predict over meshgrid
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[Y==0, 0], X[Y==0, 1], c='green', label='Healthy (0)', alpha=0.6)
    plt.scatter(X[Y==1, 0], X[Y==1, 1], c='red', label='Failure (1)', alpha=0.6, marker='x')
    acc = accuracy_score(Y, model.predict(X))
    plt.title(f"{model_name} Decision Boundary ({acc:.2%})")
    plt.xlabel("X1 = Amplifier Gain Diff")
    plt.ylabel("X2 = Span Loss Diff")
    plt.legend()
    plt.show()

def evaluate_models():
    X_train, X_test, Y_train, Y_test, scaler = load_and_prepare_data()
    if X_test is None:
        return

    for file in MODEL_FILES:
        model_path = os.path.join(MODEL_DIR, file)
        model_name = file.replace(".pkl", "")
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found. Skipping...")
            continue

        model = joblib.load(model_path)
        preds = model.predict(X_test)
        acc = accuracy_score(Y_test, preds)
        print(f"\n--- {model_name} Evaluation ---")
        print(f"Accuracy: {acc:.2%}")
        print(classification_report(Y_test, preds, target_names=['Healthy', 'Failure'], zero_division=0))

      
        plot_decision_boundary(model, X_test, Y_test, model_name)

evaluate_models()
