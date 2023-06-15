"""
train model on the ingested data
"""

import sys
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

#################Function for training the model
def train_model():
    """
    Train machine learning model on ingested data and saves the model
    """
    #load data
    logging.info("Load data")
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    y_train = df.pop('exited')
    X_train = df.drop(['corporation'], axis=1)
    #use this logistic regression for training
    logging.info("Load model")
    LR_model = LogisticRegression(
        C=1.0, class_weight=None, 
        dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
        )
    
    #fit the logistic regression to your data
    logging.info("Train model")
    LR_model.fit(X_train, y_train)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info("Saving trained model")
    pickle.dump(
        LR_model, 
        open( os.path.join(model_path, 'trainedmodel.pkl'), 'wb'))
    
if __name__ == '__main__':
    train_model()
