# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (21 September 2024)
            # Developers: Akshat Rastogi and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script implements K-Means clustering algorithm for customer segmentation. It provides functionality to train and evaluate K-Means clustering models.
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
from sklearn.cluster import KMeans  # Importing the K-Means clustering algorithm
from ingest_transform import preprocess_data, scale_data  # Custom functions for data preprocessing and scaling

# Importing helper functions from local files
# from load import load_train  # (Commented out as it is not used in this snippet)
from evaluate import evaluate_model  # Function to evaluate the clustering model's performance

def train_model(df, n, save_path):
    """Train KMeans model using DataFrame directly and save to user-specified path"""
    try:
        # Append the file name to the provided save path
        save_path = save_path + r'\kmeans.pkl'

        # Get numpy array from DataFrame
        X = df.values

        # Scale the data
        X_pca = scale_data(X)

        # Train the KMeans model
        kmeans = KMeans(n_clusters=n, random_state=42).fit(X_pca)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Evaluate the model
        evals = evaluate_model(X_pca, labels, "K-Means", centroids)

        # Save the model to the specified path
        joblib.dump(kmeans, save_path)
        print(f"Model successfully saved to: {save_path}")

        return evals

    except Exception as e:
        print(f"Error in KMeans training: {e}")
        return None

