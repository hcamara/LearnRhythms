import json
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

numpy.random.seed(7)

# Load config file
config = json.load(open("./config.json"))

# Load Training dataset
forTraining = numpy.loadtxt(config['datasets_path'] + "forTraining.txt", delimiter=",")

# Define inputs and outputs training data
inValues = forTraining[:,0:2]
outValues = forTraining[:,2]

# Neural network Layers [input dimension 2 | hidden layers 12 | sigmoid activation for classification]
model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', \
     verbose=1, save_best_only=True, mode='auto')

# Optimizer configuration for classification problem
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Model compilation and metrics definition
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training parameters configuration
model.fit(inValues, outValues, epochs=200, batch_size=10)

# Model evaluation
scores = model.evaluate(inValues, outValues)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Save models [json | hdf5]
modelJson = model.to_json()
with open(config['model_path'] + config['model_name'] + ".json", "w") as jsonFile:
    jsonFile.write(modelJson)

model.save_weights(config['model_path'] + config['model_name'] + ".h5")