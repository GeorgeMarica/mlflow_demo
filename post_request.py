import requests
import pandas as pd

# Construct request=================================================================
headers = {'Content-Type': 'application/json; format=pandas-split'}
df = pd.DataFrame([[23, 0.029, 0.48, 0.98, 8.2, 29, 3.33, 1.2, 0.39, 75, 0.66]],
                  columns=["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide",
                           "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"])
data = df.to_json(orient='split')
# Send request======================================================================
r = requests.post(url='http://d00c49c5-0354-417e-9dd9-e184f4706d0e.westus.azurecontainer.io/score', data=data, headers=headers)
print('Evaluation score is ' + r.text)
