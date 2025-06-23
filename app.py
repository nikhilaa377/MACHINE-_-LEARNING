from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('SLR_MODEL.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = float(request.form['experience'])
    prediction = model.predict(np.array([[experience]]))[0]
    prediction = round(prediction, 2)
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
