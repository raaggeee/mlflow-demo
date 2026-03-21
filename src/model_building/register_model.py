from mlflow.tracking import MlflowClient
import mlflow
import dagshub


dagshub.init(repo_owner="raaggeee", repo_name="mlflow-demo", mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/raaggeee/mlflow-demo.mlflow")

client = MlflowClient()

client.transition_model_version_stage(
    name="random-fores-model",
    version=2, stage="Production",
    archive_existing_versions=True
)