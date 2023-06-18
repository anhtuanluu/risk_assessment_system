
import os
import re
import sys
import logging
import pandas as pd
import glob
from sklearn.metrics import f1_score
import json
import scoring
import training
import ingestion
import reporting
import deployment
import diagnostics
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

def main():
    logging.info("Check new data")
    ##################Check and read new data
    #first, read ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()}
    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = set(glob.glob(
                    os.path.join(input_folder_path, "*.csv")))
    difference = ingested_files.difference(source_files)

    ##################Deciding whether to proceed, part 1
    if difference == set():
        logging.info("No new data")
        return
    logging.info("Ingest new data")
    ingestion.merge_multiple_dataframe()

    ##################Checking for model drift
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as file:
        old_score = re.findall(r'\d*\.?\d+', file.read())[1]
        old_score = float(old_score)

    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    y_test = df.pop('exited')
    X_test = df.drop(['corporation'], axis=1)
    y_pred = diagnostics.model_predictions(X_test)
    new_score = f1_score(y_test, y_pred)

    if(new_score >= old_score):
        logging.info("No model drift")
        return
    
    # Re-training
    logging.info("Re-train model")
    training.train_model()
    logging.info("Re-score model")
    scoring.score_model()

    ##################Re-deployment
    logging.info("Re-deploy model")
    deployment.store_model_into_pickle()
    ##################Diagnostics and reporting
    logging.info("Run diagnostics and reporting")
    reporting.plot_confusion_matrix()
    os.system("python apicalls.py")
if __name__ == '__main__':
    main()
