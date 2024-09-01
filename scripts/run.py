import importlib
import json
import mlflow
import os
import atexit
import shutil


setup = {
    "KNN": {
        "algorithm": "K Nearest Neighbors",
        "abbreviation": "KNN",
        "config_file": "config_KNN.json",
        "train_function": "train_and_evaluate_model_KNN",
    },
    "SVM": {
        "algorithm": "Support Vector Machine",
        "abbreviation": "SVM",
        "config_file": "config_SVM.json",
        "train_function": "train_and_evaluate_model_SVM",
    },
    "RF": {
        "algorithm": "Random Forest",
        "abbreviation": "RF",
        "config_file": "config_RF.json",
        "train_function": "train_and_evaluate_model_RF",
    },
    "LR": {
        "algorithm": "Logistic Regression",
        "abbreviation": "LR",
        "config_file": "config_LR.json",
        "train_function": "train_and_evaluate_model_LR",
    }
}


mlflow.set_tracking_uri("http://localhost:5000")
current_file_directory = os.path.dirname(os.path.abspath(__file__))
temp_folder_path = os.path.join(current_file_directory, 'temp_plots')
os.mkdir(temp_folder_path)

# This ensures that the temp folder will be deleted regardless if there is an error or not
def cleanup(current_file_directory):
    mlflow.log_artifacts(os.path.join(current_file_directory, 'temp_plots'), artifact_path="plots")
    mlflow.log_artifact(os.path.join(current_file_directory, 'log.txt'))
    
    mlflow.end_run()

    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    if os.path.exists("log.txt"):
        os.remove("log.txt")

atexit.register(cleanup, current_file_directory)

# Setting the current working directory to the directory of the run script
os.chdir(current_file_directory)

def run_step(module_name, function_name, data, params):
    print("Running", module_name, function_name)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function(data, params)

setup = setup['SVM']

# Start MLflow run
mlflow.start_run(
    experiment_id=3,
    run_name=f"{setup['abbreviation']} feature extraction: PCA",
    tags={"algorithm": setup['algorithm'], "domain": "Time Domain"},
    description=""
)

config_file = setup['config_file']

# Load configuration
with open(config_file) as f:
    config = json.load(f)

mlflow.log_artifact(os.path.join(current_file_directory, config_file))
mlflow.log_artifact(os.path.join(current_file_directory, 'run.py'), artifact_path="scripts")
mlflow.log_artifact(os.path.join(current_file_directory, 'data_load.py'), artifact_path="scripts")
mlflow.log_artifact(os.path.join(current_file_directory, 'preprocessing.py'), artifact_path="scripts")
mlflow.log_artifact(os.path.join(current_file_directory, 'feature_engineering.py'), artifact_path="scripts")
mlflow.log_artifact(os.path.join(current_file_directory, 'model_training_and_evaluation.py'), artifact_path="scripts")

data_load_params = config['data_load']['params']
for param, value in data_load_params.items():
    mlflow.log_param(param, value)

preprocess_params = config['preprocessing']['params']
for param, value in preprocess_params.items():
    mlflow.log_param(param, value)

feature_engineering_params = config['feature_engineering']['params']
for param, value in feature_engineering_params.items():
    mlflow.log_param(param, value)

model_training_params = config['model_training_and_evaluation']['params']
for param, value in model_training_params.items():
    mlflow.log_param(param, value)


# Step 1: Load data
data = run_step('data_load', 'load_data', None, config['data_load']['params'])

# Step 2: Preprocess data
X, y, groups, all_scans = run_step('preprocessing', 'preprocess_data_time_domain', data, config['preprocessing']['params'])

# Step 3: Feature engineering
X, y, groups, all_scans = run_step('feature_engineering', 'feature_engineer_simple', (X, y, groups, all_scans), config['feature_engineering']['params'])

# Step 4: Model training and evaluation
model, metrics = run_step('model_training_and_evaluation', setup['train_function'], (X, y, groups, all_scans), config['model_training_and_evaluation']['params'])

for metric, value in metrics.items():
    mlflow.log_metric(metric, value)

mlflow.sklearn.log_model(model, "model")
