from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    DataCollector,
    DeploymentCollection,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
import mlflow
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError

import argparse
import datetime


ml_client = MLClient.from_config(DefaultAzureCredential())

tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri

mlflow.set_tracking_uri(tracking_uri)

last_run = mlflow.search_runs(experiment_names = ["diabetes-training"], output_format = "list")[-1]

run_id = last_run.info.run_id

mlflow.artifacts.download_artifacts(run_id = run_id, dst_path = "model", artifact_path = "model")