from azureml.core import Workspace
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from azureml.core.webservice import AciWebservice, Webservice
import mlflow.azureml
import warnings



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
    run_id1 = "83239cfc-d51a-4574-94bd-d969eec61789"
    model_uri = "runs:/" + run_id1 + "/model"
    lr = mlflow.sklearn.load_model(model_uri)
    lr.fit(train_x, train_y)
    print(lr.predict(test_x))