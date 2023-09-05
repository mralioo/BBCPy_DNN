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


def load_mlflow_folder_via_ssh(ssh_client, remote_path, local_path):
    sftp_client = ssh_client.open_sftp()

    sftp_client.get(remote_path, local_path)
    sftp_client.close()

    return local_path


if __name__ == "__main__":
    # Replace with the path to the directory containing your copied MLflow logs
    # logs_directory = "C:/Users/alioo/Desktop/MA/bbcpy_AutoML/logs/mlflow_remote/mlflow/mlruns"
    logs_directory = "C:/Users/alioo/Desktop/MA/05092023/mlflow/mlruns"
    # ssh_client = "ali_alouane@hydra.ml.tu-berlin.de"
    remote_path = "/home/ali_alouane/bbcpy_AutoML/logs/mlflow_remote/mlflow/mlruns"

    update_meta_yaml(logs_directory)
