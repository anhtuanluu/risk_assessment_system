"""
Generate a confusion matrix
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import logging
import diagnostics
from sklearn.metrics import confusion_matrix
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 

##############Function for reporting
def plot_confusion_matrix():
    """
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """
    logging.info("Load data")
    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_test = df.pop('exited')
    X_test = df.drop(['corporation'], axis=1)

    logging.info("Test data")
    y_pred = diagnostics.model_predictions(X_test)

    logging.info("Save confusion matrix")
    confusion_matrices = confusion_matrix(y_test, y_pred)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(confusion_matrices, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    plt.title('Confusion Matrix')
    cfm_plot = sns.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig(os.path.join(model_path, "confusionmatrix.png"))

if __name__ == '__main__':
    plot_confusion_matrix()
