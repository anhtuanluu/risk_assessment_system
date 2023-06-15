"""
ingesting data
"""
import sys
import pandas as pd
import os
import json
from datetime import datetime
import logging
import glob

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe():
    """
    check for datasets, compile them together, write logs to 
    ingestedfiles.txt and write to an output file finaldata.csv
    """
    df_all = pd.DataFrame()
    file_names = []
    file_lst = glob.glob(os.path.join(input_folder_path, "*.csv"))
    logging.info(f"Read all files from {input_folder_path}")
    for file in file_lst:
        df = pd.read_csv(file)
        file_names.append(file)
        # concat all data
        df_all = pd.concat([df_all, df])

    logging.info("Clean data")
    df = df.drop_duplicates().reset_index(drop=1)

    logging.info("Save logs")
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "a") as file:
        file.write(
            f"Ingest date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nFile names:\n")
        file.write("\n".join(file_names) + '\n')

    logging.info("Write final data")
    df_all.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

if __name__ == '__main__':
    merge_multiple_dataframe()
