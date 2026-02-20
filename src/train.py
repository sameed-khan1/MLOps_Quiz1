import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(train_path, model_path):
    print(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)
    
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X, y)
    
    # Check training accuracy
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Training Accuracy: {acc:.4f}")
    
    # Save model
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Training complete!")

if __name__ == "__main__":
    train_file = "data/processed/train.csv"
    model_file = "models/logistic_model.pkl"
    train_model(train_file, model_file)
