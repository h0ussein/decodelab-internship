import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def run_data_classification_pipeline():
    print("====================================================")
    print("   DecodeLabs AI Pipeline: Project 2 - Data Class   ")
    print("====================================================\n")

    print("[1/4] Loading Iris Dataset...")
    iris = load_iris()
    X = iris.data  
    y = iris.target 
    
    print(f"-> Dataset loaded successfully.")
    print(f"-> Total samples: {X.shape[0]}, Total features: {X.shape[1]}")
    print(f"-> Target classes: {iris.target_names}\n")

    print("[2/4] Splitting dataset into Training and Testing sets (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    print(f"-> Training samples: {X_train.shape[0]}")
    print(f"-> Testing samples: {X_test.shape[0]}\n")

    print("[3/4] Applying StandardScaler (Mean=0, Variance=1)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("-> Features scaled successfully to remove order bias.\n")

    print("[4/4] Training Classification Model (Logistic Regression)...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)
    print("-> Model training complete.")

    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n================ EVALUATION REPORT ================")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("---------------------------------------------------")
    print("Classification Metrics Detailed Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("===================================================\n")

if __name__ == "__main__":
    run_data_classification_pipeline()