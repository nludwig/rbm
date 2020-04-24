import numpy as np
from numpy.random import RandomState

class RestrictedBoltzmannMachine:

    #
    #initialization methods
    #

    def __init__(self, visibleLayer, hiddenLayer,
                 temperature=1.,
                 sigma=0.01, visibleProportionOn=None,
                 parameterFile=None,
                 rng=None, rngSeed=1337):
        self.temperature = temperature
        self.beta = 1. / self.temperature

        if rng is None:
            self.rng = RandomState(seed=rngSeed)
        else:
            self.rng = rng

        self.visibleLayer = visibleLayer
        self.hiddenLayer = hiddenLayer

        if parameterFile is None:
            self.visibleBias = np.zeros(len(self.visibleLayer))
            self.initializeVisibleBias(visibleProportionOn)
            self.initializeHiddenBias()
            self.initializeWeights(sigma)
        else:
            self.loadParameterFile(parameterFile)

    def initializeVisibleBias(self, visibleProportionOn=None):
        if visibleProportionOn is None:
            self.visibleBias = np.zeros(len(self.hiddenLayer))
        else:
            visibleProportionOn[np.isclose(visibleProportionOn, 0.)] = 0.01
            visibleProportionOn[np.isclose(visibleProportionOn, 1.)] = 0.99
            self.visibleBias = np.log(visibleProportionOn / (1. - visibleProportionOn))

    def initializeHiddenBias(self):
        self.hiddenBias = np.zeros(len(self.hiddenLayer))

    def initializeWeights(self, sigma=0.01):
        self.weights = self.rng.normal(scale=sigma, size=(len(self.visibleLayer), len(self.hiddenLayer)))

    def loadParameterFile(self, parameterFile):
        #assert type(parameterFile) == file
        fileContents = [float(line.strip()) for line in parameterFile]
        visibleSlice = slice(0,
                             len(self.visibleLayer))
        hiddenSlice = slice(len(self.visibleLayer),
                            len(self.visibleLayer)+len(self.hiddenLayer))
        weightsSlice = slice(len(self.visibleLayer)+len(self.hiddenLayer),
                             len(self.visibleLayer)+len(self.hiddenLayer)
                             +len(self.visibleLayer)*len(self.hiddenLayer))
        self.visibleBias = np.array(fileContents[visibleSlice])
        self.hiddenBias = np.array(fileContents[hiddenSlice])
        self.weights = np.array(fileContents[weightsSlice]).reshape((len(self.visibleLayer), len(self.hiddenLayer)))

    def dumpParameterFile(self, parameterFile):
        #assert type(parameterFile) == file
        for theta in self.visibleBias:
            print(f'{theta}', file=parameterFile)
        for theta in self.hiddenBias:
            print(f'{theta}', file=parameterFile)
        for theta in self.weights.flatten():
            print(f'{theta}', file=parameterFile)


    #
    #prediction methods
    #

    def hiddenConditionalProbabilities(self):
        b = self.hiddenBias
        w = self.weights
        v = self.visibleLayer
        conditionalEnergy = -(b + v@w)
        return logistic(self.beta * conditionalEnergy)

    def visibleConditionalProbabilities(self):
        a = self.visibleBias
        w = self.weights
        h = self.hiddenLayer
        conditionalEnergy = -(a + w@h)
        return logistic(self.beta * conditionalEnergy)

    def rollBernoulliProbabilities(self, probabilities):
        rolls = self.rng.uniform(size=probabilities.shape)
        return (rolls <= probabilities).astype(np.float_)

    #todo: vectorize for better speed with minibatches
    def gibbsSample(self, visibleData=None, hiddenUnitsStochastic=False):
        if visibleData is not None:
            self.visibleLayer = np.copy(visibleData)
        hiddenLayerProbabilities = self.hiddenConditionalProbabilities()
        self.hiddenLayer = \
            self.rollBernoulliProbabilities(hiddenLayerProbabilities) \
            if hiddenUnitsStochastic is True else \
            hiddenLayerProbabilities
        self.visibleLayer = self.visibleConditionalProbabilities()
        return self.visibleLayer, self.hiddenLayer


    #
    #training methods
    #

    #todo: vectorize for better speed with minibatches
    def computePCDGradient(self, miniBatch, miniFantasyBatch, nCDSteps=1):
        #compute "positive"/data mean
        visibleDataMean, hiddenDataMean, weightDataMean, _ = \
            self.computePCDGradientHalves(miniBatch, True)

        #compute "negative"/model mean
        visibleModelMean, hiddenModelMean, weightModelMean, newFantasy = \
            self.computePCDGradientHalves(miniFantasyBatch, False, nCDSteps=nCDSteps, saveVisible=True)

        #compute gradients & return
        visibleGradient = visibleDataMean - visibleModelMean
        hiddenGradient = hiddenDataMean - hiddenModelMean
        weightGradient = weightDataMean - weightModelMean
        return visibleGradient, hiddenGradient, weightGradient, newFantasy

    #todo: vectorize for better speed with minibatches
    def computePCDGradientHalves(self, miniBatch, hiddenUnitsStochastic, nCDSteps=1, saveVisible=False):
        visibleMean = np.zeros(len(self.visibleLayer))
        hiddenMean = np.zeros(len(self.hiddenLayer))
        newVisible = np.zeros_like(miniBatch) if saveVisible is True else None
        for i, observation in enumerate(miniBatch):
            for _ in range(nCDSteps):
                visibleOut, hiddenOut = \
                    self.gibbsSample(visibleData=observation, hiddenUnitsStochastic=hiddenUnitsStochastic)
                observation = None #if nCDSteps > 1, use visibleData as found in previous step
            visibleMean += visibleOut
            hiddenMean += hiddenOut
            if saveVisible:
                newVisible[i] = visibleOut
        weightMean = visibleMean[:, None] * hiddenMean[None, :]
        visibleMean /= miniBatch.shape[0]
        hiddenMean /= miniBatch.shape[0]
        weightMean /= miniBatch.shape[0]
        return visibleMean, hiddenMean, weightMean, newVisible

    def sgd(self, visibleGradient, hiddenGradient, weightGradient, learningRate):
        self.visibleBias += learningRate * visibleGradient
        self.hiddenBias += learningRate * hiddenGradient
        self.weights += learningRate * weightGradient

    def updateParametersSGD(self, miniBatch, miniFantasyBatch, learningRate, nCDSteps=1):
        visibleGradient, hiddenGradient, weightGradient, newFantasy = \
            self.computePCDGradient(miniBatch, miniFantasyBatch, nCDSteps=nCDSteps)
        self.sgd(visibleGradient, hiddenGradient, weightGradient, learningRate)
        return newFantasy


    #
    #miscellaneous methods
    #

    def setRngSeed(self, rngSeed):
        self.rng.seed(rngSeed)

    def __len__(self):
        return len(self.visibleLayer), len(self.hiddenLayer)

def logistic(x):
    return 1. / (1. + np.exp(-x))
