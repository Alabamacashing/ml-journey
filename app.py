from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("titanic_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = pd.DataFrame([data])
    prediction = model.predict(features)
    return jsonify({"survived": int(prediction[0])})

if __name__ == "__main__":
   app.run(host="0.0.0.0", debug=False)