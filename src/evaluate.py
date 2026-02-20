import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

def evaluate_model(test_path, model_path, results_path):
    print(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)
    
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print("Performing evaluation...")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Final Accuracy: {acc:.4f}")
    
    # Save accuracy to results
    if not os.path.exists(os.path.dirname(results_path)):
        os.makedirs(os.path.dirname(results_path))
        
    with open(results_path, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    test_file = "data/processed/test.csv"
    model_file = "models/logistic_model.pkl"
    results_file = "results/accuracy.txt"
    evaluate_model(test_file, model_file, results_file)
