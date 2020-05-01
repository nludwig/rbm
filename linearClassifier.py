import numpy as np

class LinearClassifier:
    def __init__(self, parameters):
        self.parameters = np.copy(parameters)

    def predict(self, data):
        return data @ self.parameters
    
    def predictProbabilities(self, data):
        return logistic(self.predict(data))
    
    def computeGradientInProbabilities(self, data, labels):
        linearLabels = self.predictLinearProbabilities(data)
        gradient = ((linearLabels - labels) * linearLabels * (1. - linearLabels)) @ data
        return gradient
        #if len(data.shape) == 1:
        #    return gradient
        #else:
        #    return gradient / data.shape[0]
    
    def updateParameters(self, data, labels):
        gradient = self.computeGradientInProbabilities(data, labels)
        self.parameters += gradient

    def __len__(self):
        return len(self.parameters)

def logistic(x):
    return 1. / (1. + np.exp(-x))
