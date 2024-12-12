# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Version: V 1.1 (21 September 2024)
        # Unit test: Pass
        # Integration test: Pass
     
    # Description: This script implements Gaussian Mixture Model (GMM) clustering algorithm for customer segmentation. It provides functionality to train and evaluate GMM clustering models.
        # SQLite: Yes
        # MQs: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # numpy 1.24.3
        # pandas 1.5.3
        # scikit-learn 1.2.2
        # joblib 1.3.1

import pandas as pd  # For data manipulation and analysis
import joblib  # For saving and loading trained models
from sklearn.mixture import GaussianMixture  # Importing the Gaussian Mixture Model for clustering
from ingest_transform import preprocess_data, scale_data  # Custom functions for preprocessing and scaling data

# Importing helper functions from local files
# from load import load_train  # (Commented out as it is not used in this snippet)
from evaluate import evaluate_model  # Function to evaluate the clustering model's performance

def train_model(df, n, save_path):
    """Train GMM model using DataFrame directly and save to user-specified path"""
    try:
        # Append the file name to the provided save path
        save_path = save_path + r'gmm.pkl'

        # Convert DataFrame to numpy array
        X = df.values

        # Scale the data
        X_pca = scale_data(X)

        # Train the Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n, random_state=42)
        labels = gmm.fit_predict(X_pca)

        # Evaluate the model
        evals = evaluate_model(X_pca, labels, 'Gaussian Mixture Model')

        # Save the model to the specified path
        joblib.dump(gmm, save_path)
        print(f"Model successfully saved to: {save_path}")

        return evals

    except Exception as e:
        print(f"Error in GMM training: {e}")
        return None

