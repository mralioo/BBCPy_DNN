import mlflow

def load_haprams_run_id(run_id: str, mlflow_root_path):

    mlflow.set_tracking_uri(mlflow_root_path)
    mlflow.set_experiment("Haprams")
    run = mlflow.get_run(run_id)
    params = run.data.params
    return params