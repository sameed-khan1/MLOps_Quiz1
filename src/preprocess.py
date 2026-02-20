import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_data(input_path, output_dir):
    # Load dataset
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Handle missing values: fill numeric with mean, categorical with mode
    # For this simple dataset, feature1 and feature2 are numeric
    print("Handling missing values...")
    df['feature1'] = df['feature1'].fillna(df['feature1'].mean())
    df['feature2'] = df['feature2'].fillna(df['feature2'].mean())
    
    # Split into train/test
    print("Splitting into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save processed data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving processed data to {output_dir}...")
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    input_file = "data/raw/data.csv"
    output_folder = "data/processed"
    preprocess_data(input_file, output_folder)
