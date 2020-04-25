import numpy as np
from numpy.random import RandomState
from adam import Adam

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
            self.initializeVisibleBias(visibleProportionOn=visibleProportionOn)
            self.initializeHiddenBias()
            self.initializeWeights(sigma)
        else:
            self.loadParameterFile(parameterFile)

    def initializeVisibleBias(self, visibleProportionOn=None):
        if visibleProportionOn is None:
            self.visibleBias = np.zeros(len(self.hiddenLayer))
        else:
            visibleProportionOn[np.isclose(visibleProportionOn, 0.)] = 0.01
            #visibleProportionOn[np.isclose(visibleProportionOn, 1.)] = 0.99
            #self.visibleBias = np.log(visibleProportionOn / (1. - visibleProportionOn))
            self.visibleBias = 1. / visibleProportionOn

    def initializeHiddenBias(self):
        self.hiddenBias = np.zeros(len(self.hiddenLayer))

    def initializeWeights(self, sigma=0.01):
        self.weights = self.rng.normal(scale=sigma, size=(len(self.visibleLayer), len(self.hiddenLayer)))

    def loadParameterFile(self, parameterFile):
        #assert type(parameterFile) == file
        fileContents = [float(line.strip()) for line in parameterFile]
        visibleSlice = slice(0, len(self.visibleLayer))
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
        #compute unit energies and activation probabilities
        conditionalEnergies = -(self.hiddenBias + self.visibleLayer@self.weights)
        return logistic(self.beta * conditionalEnergies)

    def visibleConditionalProbabilities(self):
        #compute unit energies and activation probabilities
        conditionalEnergies = -(self.visibleBias + self.weights@self.hiddenLayer)
        return logistic(self.beta * conditionalEnergies)

    def rollBernoulliProbabilities(self, probabilities):
        rolls = self.rng.uniform(size=probabilities.shape)
        return (rolls < probabilities).astype(np.float_)

    #todo: vectorize for better speed with minibatches
    def gibbsSample(self, visibleData=None, hiddenUnitsStochastic=False):
        #load visibleDate; else use current values
        if visibleData is not None:
            self.visibleLayer = np.copy(visibleData)
        #compute hidden activation probabilities given visible
        hiddenLayerProbabilities = self.hiddenConditionalProbabilities()
        self.hiddenLayer = \
            self.rollBernoulliProbabilities(hiddenLayerProbabilities) \
            if hiddenUnitsStochastic is True else \
            hiddenLayerProbabilities
        #compute visible activation probabilities given hidden
        visibleLayerProbabilities = self.visibleConditionalProbabilities()
        self.visibleLayer = visibleLayerProbabilities
        return visibleLayerProbabilities, hiddenLayerProbabilities


    #
    #training methods
    #

    #todo: vectorize for better speed with minibatches
    def computePCDGradient(self, miniBatch, miniFantasyBatch, nCDSteps=1, l2Coefficient=0.):
        #compute "positive"/data mean
        visibleDataMean, hiddenDataMean, weightDataMean, _ = \
            self.computePCDGradientHalves(miniBatch, True, clamp=True)

        #compute "negative"/model mean
        visibleModelMean, hiddenModelMean, weightModelMean, newFantasy = \
            self.computePCDGradientHalves(miniFantasyBatch, False, nCDSteps=nCDSteps, saveVisible=True)

        #compute gradients & return
        visibleGradient = visibleDataMean - visibleModelMean
        hiddenGradient = hiddenDataMean - hiddenModelMean
        weightGradient = weightDataMean - weightModelMean - l2Coefficient * self.weights
        return visibleGradient, hiddenGradient, weightGradient, newFantasy

    #todo: vectorize for better speed with minibatches
    def computePCDGradientHalves(self, miniBatch, hiddenUnitsStochastic, nCDSteps=1, saveVisible=False, clamp=False):
        visibleMean = np.zeros(len(self.visibleLayer))
        hiddenMean = np.zeros(len(self.hiddenLayer))
        newVisible = np.zeros_like(miniBatch) if saveVisible is True else None
        for i, observation in enumerate(miniBatch):
            visibleIn = np.copy(observation)
            for _ in range(nCDSteps):
                visibleOut, hiddenOut = \
                    self.gibbsSample(visibleData=visibleIn, hiddenUnitsStochastic=hiddenUnitsStochastic)
                visibleIn = None #if nCDSteps > 1, use visibleData as found in previous step
            visibleMean += observation if clamp is True else visibleOut
            hiddenMean += hiddenOut
            if saveVisible is True:
                newVisible[i] = visibleOut
        visibleMean /= miniBatch.shape[0]
        hiddenMean /= miniBatch.shape[0]
        weightMean = visibleMean[:, None] * hiddenMean[None, :]
        return visibleMean, hiddenMean, weightMean, newVisible

    def sgd(self, visibleGradient, hiddenGradient, weightGradient, learningRate):
        self.visibleBias -= learningRate * visibleGradient
        self.hiddenBias -= learningRate * hiddenGradient
        self.weights -= learningRate * weightGradient

    def updateParametersSGD(self, miniBatch, miniFantasyBatch, learningRate, nCDSteps=1, l2Coefficient=0., verbose=False):
        visibleGradient, hiddenGradient, weightGradient, newFantasy = \
            self.computePCDGradient(miniBatch, miniFantasyBatch, nCDSteps=nCDSteps, l2Coefficient=l2Coefficient)
        self.sgd(visibleGradient, hiddenGradient, weightGradient, learningRate)
        if verbose is True:
            print('{:.3f}\t{:.3f}\t{:.3f}'.format(visibleGradient.mean(),
                                                  hiddenGradient.mean(),
                                                  weightGradient.mean()))
        return newFantasy

    def updateParametersAdam(self, miniBatch, miniFantasyBatch, adams, nCDSteps=1, l2Coefficient=0., verbose=False):
        #compute gradients
        visibleGradient, hiddenGradient, weightGradient, newFantasy = \
            self.computePCDGradient(miniBatch, miniFantasyBatch, nCDSteps=nCDSteps, l2Coefficient=l2Coefficient)
        #compute adam updates
        visibleStep = adams['visible'].computeAdamStep(visibleGradient)
        hiddenStep = adams['hidden'].computeAdamStep(hiddenGradient)
        weightStep = adams['weights'].computeAdamStep(weightGradient)
        #update parameters
        self.visibleBias -= visibleStep
        self.hiddenBias -= hiddenStep
        self.weights -= weightStep

        if verbose is True:
            print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                                          visibleGradient.mean(),
                                          hiddenGradient.mean(),
                                          weightGradient.mean(),
                                          visibleStep.mean(),
                                          hiddenStep.mean(),
                                          weightStep.mean()))
        return newFantasy

    def computeReconstructionError(self, miniBatch, nCDSteps=1):
        meanSquaredError = 0.
        for i, observation in enumerate(miniBatch):
            visibleIn = np.copy(observation)
            for _ in range(nCDSteps):
                visibleOut, _ = self.gibbsSample(visibleData=visibleIn)
                visibleIn = None #if nCDSteps > 1, use visibleData as found in previous step
            #visibleData = self.rollBernoulliProbabilities(visibleData)
            sampleError = observation - visibleOut
            meanSquaredError += (sampleError * sampleError).mean()
        return meanSquaredError


    #
    #miscellaneous methods
    #

    def setRngSeed(self, rngSeed):
        self.rng.seed(rngSeed)

    def __len__(self):
        return len(self.visibleLayer), len(self.hiddenLayer)

def logistic(x):
    return 1. / (1. + np.exp(-x))
        
#def logistic(x):
#    np.seterr(all='raise')
#    try:
#        value = 1. / (1. + np.exp(-x))
#    except FloatingPointError:
#        for y in x:
#            try:
#                v = 1./(1.+np.exp(-y))
#            except FloatingPointError:
#                np.seterr(all='ignore')
#                v = 1./(1.+np.exp(-y))
#                print('{} {} done'.format(y, v), flush=True)
#                exit(1)
#            else:
#                print(y, v, flush=True)
#    else:
#        return value
