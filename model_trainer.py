import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from data_processor import load_and_prepare_data, RANDOMSTATE

def train_and_save_models():
    X_train, X_test, Y_train, Y_test, scaler = load_and_prepare_data()

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            random_state=RANDOMSTATE, class_weight='balanced'),
        "LogisticRegression": LogisticRegression(
            solver="liblinear", class_weight='balanced', random_state=RANDOMSTATE),
        "RBFSVM": SVC(
            kernel='rbf', gamma='scale', C=1.0, class_weight='balanced', random_state=RANDOMSTATE),
        "QDA": QDA()
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, Y_train)
        joblib.dump(model, f"models/{name}.pkl")
        print(f"Saved {name} to models/{name}.pkl")

    joblib.dump(scaler, "models/scaler.pkl")
    print("All models and scaler saved successfully!")

if __name__ == "__main__":
    train_and_save_models()
