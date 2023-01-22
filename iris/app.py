import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

model = pickle.load(open('model.pkl', 'rb')) # load the model that  have been saved using pickle command.

# Creating our app
app = Flask(__name__)
@app.route('/')   # signifies what to do when the browser hit the particular URL
def home():
      return render_template('index.html')


@app.route('/predict',methods=['POST']) #we wrote a function that returns predicted value when we browse to /predict.
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()] #all the input from our HTML form and convert it into float.
   
    final_features = [np.array(int_features)] # covert them into NumPy array. So that we can directly predict the value.
    prediction = model.predict(final_features) #the loaded model helps in prediction

    output =prediction[0]  #save the output as a string
     # return the output to index.html
    return render_template('ok1.html', prediction_text='The Flower is {}'.format(output))

if __name__ == "__main__": #to execute code only if the file was run directly, and not imported.
    #  run the app inside local devlop server.
    app.run(debug=True)