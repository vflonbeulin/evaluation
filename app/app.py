from flask import Flask, render_template, request
import pickle
from sklearn.pipeline import Pipeline
import pandas as pd

app = Flask(__name__)

# Chargement du modele regression
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    global n, pipe

    area = request.args.get("area")
    sex = request.args.get("sex")
    dimension1 = int(request.args.get("dimension_1_mm"))
    dimension2 = int(request.args.get("dimension_2_mm"))
    dimension3 = int(request.args.get("dimension_3_mm"))
    mass = int(request.args.get("mass_g"))
    
    dic = {
        'area': area,
        'dimension_1_mm':dimension1,
        'dimension_2_mm':dimension2,
        'dimension_3_mm':dimension3,
        'mass_g':mass,
        'sex':sex
    }
    df_predict = pd.DataFrame(dic, index=[0])
    y = model.predict(df_predict)

    if y == 0:
        strPrediction = 'Chinensis'
    elif y == 1:
        strPrediction = 'Peale'
    else:
        strPrediction = 'Tropicalis'

    result = {"variety": strPrediction}
    return result


if __name__ == "__main__":
    app.run(debug=False)
