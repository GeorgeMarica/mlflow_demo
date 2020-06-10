from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
import mlflow.azureml

ws = Workspace.from_config('config_devtalks.json')
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# Creating Image
run_id1 = "2936f7ac-6a29-4e38-a12f-540ecac0e7af"
model_uri = "runs:/" + run_id1 + "/model"
model_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri, workspace=ws, model_name="model_wine_dev_talks",
                                                      image_name="model",description="wine_quality_model_devtalks", synchronous=False)
model_image.wait_for_creation(show_output=True)

# Create micro service
dev_webservice_name = "elasticnet"
dev_webservice_deployment_config = AciWebservice.deploy_configuration()
dev_webservice = Webservice.deploy_from_image(name=dev_webservice_name, image=model_image,
                                              deployment_config=dev_webservice_deployment_config, workspace=ws)
dev_webservice.wait_for_deployment()
