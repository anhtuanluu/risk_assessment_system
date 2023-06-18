# Dynamic Risk Assessment System

The project is to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's clients. Also set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

## Prerequisites
- Python 3

## Installation
```bash
pip install -r requirements.txt
```

## Steps
<img src="assests/fullprocess.jpg" width=700 height=300>

## Usage

- Edit config.json file
```bash
"input_folder_path": "sourcedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "models", 
"prod_deployment_path": "production_deployment"
```
-  Data ingestion
```python
python ingestion.py
```

- Train model, scoring and deployment
```python
python training.py
python scoring.py
python deployment.py
```

- Diagnostics and reporting
```python
python diagnostics.py
python reporting.py
```

- Run app
```python
python app.py
```

- Test API
```python
python apicalls.py
```

- Test full processes
```python
python fullprocess.py
```

- Start cron job
In the command line of your workspace, run the following command: service cron start  
Open your workspace's crontab file by running crontab -e in your workspace's command line. Your workspace may ask you which text editor you want to use to edit the crontab file. You can select option 3, which corresponds to the "vim" text editor.  
When you're using vim to edit the crontab, you need to press the "i" key to be able to insert a cron job.  
After you write the cron job in the crontab file, you can save your work and exit vim by pressing the escape key, and then typing ":wq" , and then press Enter. This will save your one-line cron job to the crontab file and return you to the command line. If you want to view your crontab file after exiting vim, you can run crontab -l on the command line.  