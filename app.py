from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

scaler = joblib.load('models/scaler.joblib')
model = joblib.load('models/selectkbest_Logistic_Regression_model.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    scaled_features = scaler.transform(final_features)   
    prediction = model.predict(scaled_features)
   # y_probabilities_test = model.predict_proba(final_features)
   # y_prob_success = y_probabilities_test[:, 1]
   # print("final features",final_features)
   # print("prediction:",prediction)
    output = round(prediction[0], 2)
   # y_prob=round(y_prob_success[0], 3)
   # print(output)



    if output == 0:
        output = 'Benign'
        return render_template('index.html', prediction_text=f'The predicted diagnosis of the tumor based on the input features is {output}')
    else:
         output = 'Malignant'
         return render_template('index.html', prediction_text=f'The predicted diagnosis of the tumor based on the input features is {output}')

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
