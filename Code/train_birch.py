# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Shubh Gupta
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (21 September 2024)
            # Developers: Shubh Gupta
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script implements BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) algorithm for customer segmentation. It provides functionality to train and evaluate BIRCH clustering models.
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
from sklearn.cluster import Birch  # BIRCH clustering algorithm
from ingest_transform import preprocess_data, scale_data  # Custom preprocessing and scaling functions

# Importing helper functions from the local .py files
# from load import load_train # (Commented out as it is not used in this snippet)
from evaluate import evaluate_model  # Custom function to evaluate the model's performance

import streamlit as st
from sklearn.cluster import Birch
import joblib

def train_model(df, n_cluster, threas, save_path, minmax):
    """Train BIRCH model using DataFrame directly and save to user-specified path"""
    try:
        # User input for the model save path
        # save_path = st.text_input("Enter the path to save the trained BIRCH model:", "Code/saved_model")
        save_path = save_path + r'\birch.pkl'

        # Convert DataFrame to numpy array
        X = df.values
        
        # Scale the data
        X_pca = scale_data(X, minm=minmax)
        
        # Train the BIRCH model
        birch = Birch(n_clusters=n_cluster, threshold=threas)
        labels = birch.fit_predict(X_pca)
        
        # Evaluate the model
        evals = evaluate_model(X_pca, labels, 'BIRCH')
        
        # Save the model to the specified path
        joblib.dump(birch, save_path)
        st.success(f"Model successfully saved to: {save_path}")
        
        return evals

    except Exception as e:
        st.error(f"Error in BIRCH training: {e}")
        return None

