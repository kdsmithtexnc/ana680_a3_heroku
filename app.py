from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os
app = Flask(__name__, static_url_path='/static')
filename = 'file_wine.pkl'
#model = pickle.load(open(filename, 'rb'))
rf = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    alcohol = request.form['alcohol']
    density = request.form['density']
    volatile_acidity = request.form['volatile_acidity']
    
      
    pred = rf.predict(np.array([[alcohol, density, volatile_acidity ]]))
    print(pred)
    return render_template('index.html', predict = str(pred))


if __name__ == '__main__':
     port = os.environ.get("PORT", 5000)
     app.run(debug=False, host='0.0.0.0', port=port)
