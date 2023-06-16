"""
Create API
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics
import scoring


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

# with open('config.json','r') as f:
#     config = json.load(f) 

# dataset_csv_path = os.path.join(config['output_folder_path']) 

# prediction_model = None

######################Hello
@app.route('/')
def index():
    return "Hello"

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    """
    call the prediction function
    Output: 
        json: model prediction
    """  
    file_path = request.get_json()['filepath']
    df = pd.read_csv(file_path)
    df = df.drop(['corporation', 'exited'], axis=1)
    result = diagnostics.model_predictions(df).tolist()

    return jsonify(result)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    
    """
    #check the score of the deployed model
    Output:
        str: f1 score
    """
    f1_score = scoring.score_model()
    return str(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    """
    check means, medians, and modes for each column
    Output:
        json: summary statistics
    """
    result = diagnostics.dataframe_summary()
    
    return jsonify(result)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostic():
    """
    check timing, percent NA values and dependency 
    Output:
        dict: percent NA, execution time and outdated packages
    """
    percentage = diagnostics.missing_percentage()
    timing = diagnostics.execution_time()
    dependency = diagnostics.outdated_packages_list()

    return jsonify({
                    'missing_percentage': percentage,
                    'execution_time': timing,
                    'outdated_packages': dependency
                })

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
