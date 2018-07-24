import librosa
import math

class Song(object):
    def __init__(self, dataSetPath):
        self.dataSetFile = dataSetPath

    def getSongSignature(self,music):
        y, sr = librosa.load(music,sr=None)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        arithArray, statDico, totalX, totalY = [[],{},0.0,0.0]

        # Generate statistic suite
        for key, value in enumerate(beats):
            if value < beats[len(beats)-1]:
                arithArray.append(beats[key+1] - beats[key])

        # Generate Xi, Ni from suite
        for entry in arithArray:
            if entry not in statDico:
                statDico[entry] = 1
            else:
                statDico[entry] += 1

        # Moyenne calcul
        for key in statDico:
            totalX += key*statDico[key]
            totalY += statDico[key]
        beats_moy = math.floor(totalX/totalY)

        return [tempo, beats_moy]

    def saveSong(self, signature, genre):
        # Concat signature and music genre
        status = 'init'
        if genre is not None:
            signature.append(genre)
        with open(self.dataSetFile, 'a') as dataset:
            tmpStr = ''
            for entry, value in enumerate(signature):
                tmpStr += str(value) + ','
            dataset.write(tmpStr[:len(tmpStr)-1] + '\n')
            status = 'success'
        return status
    

 


