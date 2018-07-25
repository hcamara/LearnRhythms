import os
import json
import numpy
from os import environ
from flask import Flask, render_template
from flask_restful import Resource, Api
from flask_socketio import SocketIO, emit, disconnect
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from libs.SongProcessing import Song
    
app = Flask(__name__)
api = Api(app)

# Load config file
config = json.load(open("./config.json"))
numpy.random.seed(7)

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # Load data for model prediction
    forPredictionData = numpy.loadtxt(config['datasets_path'] + 'forTesting.txt', delimiter=",")
    inValues = forPredictionData[:,0:2]
    
    # Load pretrained model
    jsonFile = open(config['model_path'] + config['model_name'] + '.json', 'r')
    pretrained_model = jsonFile.read()
    jsonFile.close()
    model = model_from_json(pretrained_model)

    # Compilation process
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Prediction process
    predictions = model.predict(inValues)
    predicted = [round(x[0]) for x in predictions]

    return render_template('api.html', data=predicted)

if __name__ == '__main__':
    port = int(environ.get('PORT',  config['port']))
    app.run(host='0.0.0.0', port=port)