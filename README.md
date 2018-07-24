# LearnFromRythm
This project consists of learning from rhythm. CNN is used for classification problem, 
tempo feature and average of beats for musical genre recognition problem.

## Getting started
LearnFromRythm is based on musical signature recognition and CNN classification.
For musical signature  tempo feature and average of beats are used to detect similarity 
between two or more songs. 

### How to generate datasets
To create a new model, you must generate training datasets.
To generate basic data sets for the tests, use the following command:

```
python datasets_generator.py
```
This command has two characteristics, first it generates sets of training data and
second, it also generates test data sets

### How to create model
To create a template and evaluate it, you must use the following command:

```
python create_model.py
```

### How to make prediction
Make the prediction is very easy, you can use the next
command to predict the genre of the song based on your previously created model

```
python predict.py
```

