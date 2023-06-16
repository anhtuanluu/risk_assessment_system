import requests
import os
import sys
import logging
import requests
import json
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

logging.info(f"Post request /prediction")
response1 = requests.post(
    f'{URL}/prediction',
    json={'filepath': os.path.join(test_data_path, 'testdata.csv')}).text

logging.info("Get scoring")
response2 = requests.get(f'{URL}/scoring').text

logging.info("Get summarystats")
response3 = requests.get(f'{URL}/summarystats').text

logging.info("Get diagnostics")
response4 = requests.get(f'{URL}/diagnostics').text

logging.info("Save report")
with open(os.path.join(model_path, 'apireturns.txt'), 'w') as file:
    file.write('Predictions:\n')
    file.write(response1)
    file.write('Score:\n')
    file.write(response2)
    file.write('\nSummarystats:\n')
    file.write(response3)
    file.write('Diagnostics:\n')
    file.write(response4)
