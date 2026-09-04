import requests

response = requests.post(
    "http://127.0.0.1:5001/predict",
    json={"Pclass": 3, "Age": 35, "Fare": 7, "Sex": 0}
)
