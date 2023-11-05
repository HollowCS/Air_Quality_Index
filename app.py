from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data = CustomData(
            City=str(request.form.get("City")),
            Date=str(request.form.get("Date")),
            PM2DOT5=float(request.form.get("PM2.5")),
            PM10=float(request.form.get("PM10")),
            NO=float(request.form.get("NO")),
            NO2=float(request.form.get("NO2")),
            NOx=float(request.form.get("NOx")),
            NH3=float(request.form.get("NH3")),
            CO=float(request.form.get("CO")),
            SO2=float(request.form.get("SO2")),
            O3=float(request.form.get("O3")),
            Benzene=float(request.form.get("Benzene")),
            Toluene=float(request.form.get("Toluene")),
            Xylene=float(request.form.get("Xylene"))
        )
        final_dataframe = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        predict = predict_pipeline.predict(final_dataframe)
        results = round(predict[0], 2)
        return render_template("results.html", final_result=results)


if __name__ == "__main__":
    app.run(
        host="localhost",
        port=4000,
        debug=True
    )
