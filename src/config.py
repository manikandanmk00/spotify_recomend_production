import os
import mlflow

def configure_mlflow():
    # Set MLflow tracking URI (local or remote)
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    
    # Optional: set experiment name
    mlflow.set_experiment("content_based_recommender")

    # Optional: set global tags
    mlflow.set_tag("project", "spotify-recommender")
