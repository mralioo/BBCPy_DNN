import os
import yaml

def update_yaml_file(file_path, root):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    # Modify the artifact_uri for run-level meta.yaml
    if 'artifact_uri' in data:
        data['artifact_uri'] = 'file://' + os.path.join(root, 'artifacts')

    # Modify the artifact_location for experiment-level meta.yaml
    if 'artifact_location' in data:
        data['artifact_location'] = 'file://' + os.path.join(root, 'artifacts')

    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)

def update_meta_yaml(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "meta.yaml":
                update_yaml_file(os.path.join(root, file), root)

if __name__ == "__main__":
    # Replace with the path to the directory containing your copied MLflow logs
    logs_directory = '/mnt/0E0823E30823C893/MA/code/bbcpy_AutoML/logs/mlflow_remote/mlflow/mlruns'
    update_meta_yaml(logs_directory)
