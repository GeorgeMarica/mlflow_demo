import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn
import mlflow
from azureml.core import Workspace

ws = Workspace.from_config('config_devtalks.json')
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

data_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# Model evaluation====================================================================================
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    csv_url = data_path
    data = pd.read_csv(csv_url, sep=';')
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    experiment_name = 'live_demo_2'
    mlflow.set_experiment(experiment_name)
    # Iterate through parameters and run experiment=====================================================
    for alpha in np.linspace(0.1, 0.3, 5).tolist():
        for l1_ratio in np.linspace(0.1, 0.3, 5).tolist():
            with mlflow.start_run():
                lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                lr.fit(train_x, train_y)
                predicted_qualities = lr.predict(test_x)
                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    # Log parameters and metrics to remote Azure server==================================================
                mlflow.log_metric("alpha", alpha)
                mlflow.log_metric("l1_ratio", l1_ratio)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                mlflow.sklearn.log_model(lr, "model")