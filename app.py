from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import math

application = Flask(__name__)
app = application

@app.route("/",methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template("predict.html")
    else:
        data = CustomData(
            request.form.get("date"),
            request.form.get("hour"),
            request.form.get("holiday"),
            request.form.get("weathersit"),
            request.form.get("temp"),
            request.form.get("humidity"),
            request.form.get("windspeed")
        )

        pred_df = data.get_data_as_data_frame()
        # print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        return render_template("predict.html",casual=math.ceil(results[0][0]),registered=math.ceil(results[1][0]))




if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)