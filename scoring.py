"""
score the model
"""
import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score
import json
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model():
    """
    this function take a trained model, load test data, 
    and calculate an F1 score for the model relative to the test data
    it write the result to the latestscore.txt file
    """
    logging.info("Load test data")
    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    logging.info("Load model")
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as file:
        LR_model = pickle.load(file)

    logging.info("Preparing test data")
    y_test = df.pop('exited')
    X_test = df.drop(['corporation'], axis=1)

    logging.info("Predicting test data")
    y_pred = LR_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(f"F1 score = {score}")

    logging.info("Save score")
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"F1 score = {score}")
    return 0
if __name__ == '__main__':
    score_model()
