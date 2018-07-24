import numpy
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# Load config file
config = json.load(open("./config.json"))
numpy.random.seed(7)

# Load data for model prediction testing
forTesting = numpy.loadtxt(config['datasets_path'] + "forTesting.txt", delimiter=",")
inValues = forTesting[:,0:2]

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
print(predicted)
