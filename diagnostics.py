
import pandas as pd
import numpy as np
import timeit
import os
import json
import sys
import logging
import pickle
import subprocess
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_production_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(df):
    """
    read the deployed model and a test dataset, calculate predictions
    Input:
        df: pandas.DataFrame
        Dataframe
    Output:
        y_pred: 
        Model predictions
    """
    logging.info("Load production model")
    with open(os.path.join(model_production_path, 'trainedmodel.pkl'),
            'rb') as file:
        LR_model = pickle.load(file)
        

    logging.info("Predict data")
    y_pred = LR_model.predict(df)
    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    # here
    """
    calculate summary statistics
    Output:
        dict: statistics
    """
    logging.info("Load final data")
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    df = df.drop(['exited'], axis=1)
    df = df.select_dtypes(include=np.number)

    logging.info("Calculating statistics for data")
    statistics_result = dict()
    for col in df.columns:
        statistics_result[col] = {
                                'mean': df[col].mean(), 
                                'median': df[col].median(), 
                                'std': df[col].std()
                                 }
    return statistics_result

def missing_percentage():
    """
    Calculate percentage of missing data

    Output:
        dict: Name and missing percentage
    """
    logging.info("Load data")
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    logging.info("Calculate missing percentage")
    missing_result = dict()
    missing_result = (df.isna().sum() / len(df) * 100).to_dict()

    return missing_result

##################Function to get timings
def execution_time():
    """
    calculate timing of training.py and ingestion.py
    Output:
        dict: execution times for each script
    """
    result = dict()
    logging.info("Calculate time for ingestion.py")
    start_time = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    ingestion_time = timeit.default_timer() - start_time

    logging.info("Calculate time for training.py")
    start_time = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'], capture_output=True)
    training_time = timeit.default_timer() - start_time

    result = {
            'ingestion_time': ingestion_time,
            'training_time': training_time
             }
    return result

##################Function to check dependencies
def outdated_packages_list():
    """
    Check dependencies status outdated

    Output:
        str: pip-outdated command
    """
    logging.info("Check dependencies")
    dependencies = subprocess.run(
        'pip-outdated requirements.txt',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8')

    result = dependencies.stdout[70:]
    return result

if __name__ == '__main__':
    logging.info("Load testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = test_df.drop(['corporation', 'exited'], axis=1)
    print("Model predictions on test data:\n", model_predictions(X_test))
    print("Summary statistics:\n", dataframe_summary())
    print("Missing percentage:\n", missing_percentage())
    print("Execution time:\n", execution_time())
    print("Outdated Packages:\n", outdated_packages_list())
