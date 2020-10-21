import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def spb_predictor():
    int_features = [int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction =int( model.predict(final_features)[0][0])
    return render_template('index.html',prediction_text='Your Normal Systolic Blood Pressure Should be {}'.format(prediction))
if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=8080)
