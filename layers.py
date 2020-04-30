import numpy as np
from numpy.random import RandomState
from adam import Adam

class RestrictedBoltzmannMachine:

    #
    #initialization methods
    #

    def __init__(self,
                 visibleLayer,
                 hiddenLayer,
                 temperature=1.,
                 sigma=0.01,
                 visibleProportionOn=None,
                 parameterFile=None,
                 rng=None,
                 rngState=None,
                 rngSeed=1337):
        self.visibleLayer = visibleLayer
        self.hiddenLayer = hiddenLayer
        self.temperature = temperature
        self.beta = 1. / self.temperature

        if rng is None:
            self.rng = RandomState(seed=rngSeed)
            if rngState is not None:
                self.rng.set_state(rngState)
        else:
            self.rng = rng

        if parameterFile is None:
            self.initializeVisibleBias(visibleProportionOn=visibleProportionOn)
            self.initializeHiddenBias()
            self.initializeWeights(sigma)
        else:
            self.loadParameterFile(parameterFile)
        self.visibleStep = np.zeros_like(self.visibleBias)
        self.hiddenStep = np.zeros_like(self.hiddenBias)
        self.weightStep = np.zeros_like(self.weights)

    def initializeVisibleBias(self, visibleProportionOn=None):
        if visibleProportionOn is None:
            self.visibleBias = np.zeros(self.visibleLayer.shape[-1])
        else:
            #find minimum non-zero value
            nonZeroMin = visibleProportionOn[visibleProportionOn > 0.].min()
            visibleProportionOn[np.isclose(visibleProportionOn, 0.)] = nonZeroMin + (0. - nonZeroMin) / 2.
            nonOneMax = visibleProportionOn[visibleProportionOn < 1.].max()
            print(f'nonZeroMin, nonOneMax: {nonZeroMin}, {nonOneMax}')
            visibleProportionOn[np.isclose(visibleProportionOn, 1.)] = nonOneMax + (1. - nonOneMax) / 2.
            self.visibleBias = np.log(visibleProportionOn / (1. - visibleProportionOn))
            #self.visibleBias = 1. / visibleProportionOn

    def initializeHiddenBias(self):
        self.hiddenBias = np.zeros(self.hiddenLayer.shape[-1])

    def initializeWeights(self, sigma=0.01):
        self.weights = self.rng.normal(scale=sigma, size=(self.visibleLayer.shape[-1], self.hiddenLayer.shape[-1]))

    def loadParameterFile(self, parameterFile):
        lv = self.visibleLayer.shape[-1]
        lh = self.hiddenLayer.shape[-1]
        visibleSlice = slice(0, lv)
        hiddenSlice = slice(lv, lv+lh)
        weightsSlice = slice(lv+lh, lv+lh+lv*lh)
        fileContents = [float(line.strip()) for line in parameterFile]
        self.visibleBias = np.array(fileContents[visibleSlice])
        self.hiddenBias = np.array(fileContents[hiddenSlice])
        self.weights = np.array(fileContents[weightsSlice]).reshape((lv, lh))

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
        conditionalEnergies = self.hiddenBias + self.visibleLayer@self.weights
        return logistic(self.beta * conditionalEnergies)

    def visibleConditionalProbabilities(self):
        conditionalEnergies = self.visibleBias + self.hiddenLayer@self.weights.T
        return logistic(self.beta * conditionalEnergies)

    def rollBernoulliProbabilities(self, probabilities):
        rolls = self.rng.uniform(size=probabilities.shape)
        return (rolls < probabilities).astype(np.float_)

    def gibbsSample(self, hiddenUnitsStochastic=False):
        #compute hidden activation probabilities given visible
        hiddenLayerProbabilities = self.hiddenConditionalProbabilities()
        if hiddenUnitsStochastic:
            self.hiddenLayer = self.rollBernoulliProbabilities(hiddenLayerProbabilities)
        else:
            self.hiddenLayer = hiddenLayerProbabilities
        #compute visible activation probabilities given hidden
        self.visibleLayer = self.visibleConditionalProbabilities()
        return self.visibleLayer, hiddenLayerProbabilities


    #
    #training methods
    #

    def computePCDGradient(self, miniBatch, miniFantasyBatch, nCDSteps=1, l1Coefficient=None, l2Coefficient=None):
        visibleDataMean, hiddenDataMean, weightDataMean = self.computePCDGradientPositiveHalf(miniBatch)
        visibleModelMean, hiddenModelMean, weightModelMean, newFantasy = \
            self.computePCDGradientNegativeHalf(miniFantasyBatch, nCDSteps=nCDSteps)

        #compute gradients & return
        visibleGradient = visibleDataMean - visibleModelMean
        hiddenGradient = hiddenDataMean - hiddenModelMean
        weightGradient = weightDataMean - weightModelMean
        if l1Coefficient is not None:
            weightGradient -= l1Coefficient * np.sign(self.weights)
        if l2Coefficient is not None:
            weightGradient -= l2Coefficient * self.weights
        return visibleGradient, hiddenGradient, weightGradient, newFantasy

    def computePCDGradientPositiveHalf(self, miniBatch):
        self.visibleLayer = miniBatch
        hiddenLayerProbabilities = self.hiddenConditionalProbabilities()
        return self.computeParameterMeans(miniBatch, hiddenLayerProbabilities)

    def computePCDGradientNegativeHalf(self, miniFantasyBatch, nCDSteps=1):
        self.visibleLayer = miniFantasyBatch
        for _ in range(nCDSteps):
            visibleOut, hiddenOut = self.gibbsSample()
        return self.computeParameterMeans(visibleOut, hiddenOut) + (visibleOut,)

    def computeParameterMeans(self, visible, hidden):
        visibleMean = visible.mean(axis=0)
        hiddenMean = hidden.mean(axis=0)
        weightMean = visibleMean[..., :, None] * hiddenMean[..., None, :] * visible.shape[0]
        return visibleMean, hiddenMean, weightMean

    def updateParameters(self):
        self.visibleBias += self.visibleStep
        self.hiddenBias +=  self.hiddenStep
        self.weights += self.weightStep

    def updateParametersSGD(self, miniBatch, miniFantasyBatch, learningRate, nCDSteps=1,
                            l1Coefficient=None, l2Coefficient=None, verbose=False):
        visibleGradient, hiddenGradient, weightGradient, newFantasy = \
            self.computePCDGradient(miniBatch, miniFantasyBatch, nCDSteps=nCDSteps,
                                    l1Coefficient=l1Coefficient, l2Coefficient=l2Coefficient)
        #hack to stop changing the *Step pointer; req'd for
        # current implementation of histograms of *Steps
        self.visibleStep += learningRate * visibleGradient - self.visibleStep
        self.hiddenStep += learningRate * hiddenGradient - self.hiddenStep
        self.weightStep += learningRate * weightGradient - self.weightStep
        self.updateParameters()
        if verbose is True:
            print('{:.3f}\t{:.3f}\t{:.3f}'.format(self.visibleStep.mean(),
                                                  self.hiddenStep.mean(),
                                                  self.weightStep.mean()))
        return newFantasy

    def updateParametersAdam(self, miniBatch, miniFantasyBatch, adams, nCDSteps=1,
                             l1Coefficient=None, l2Coefficient=None, verbose=False):
        visibleGradient, hiddenGradient, weightGradient, newFantasy = \
            self.computePCDGradient(miniBatch, miniFantasyBatch, nCDSteps=nCDSteps,
                                    l1Coefficient=l1Coefficient, l2Coefficient=l2Coefficient)
        #hack to stop changing the *Step pointer; req'd for
        # current implementation of histograms of *Steps
        self.visibleStep += adams['visible'].computeAdamStep(visibleGradient) - self.visibleStep
        self.hiddenStep += adams['hidden'].computeAdamStep(hiddenGradient) - self.hiddenStep
        self.weightStep += adams['weights'].computeAdamStep(weightGradient) - self.weightStep
        self.updateParameters()
        if verbose is True:
            print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                                          visibleGradient.mean(),
                                          hiddenGradient.mean(),
                                          weightGradient.mean(),
                                          self.visibleStep.mean(),
                                          self.hiddenStep.mean(),
                                          self.weightStep.mean()))
        return newFantasy


    #
    #analysis methods
    #

    def computeReconstructionError(self, miniBatch, nCDSteps=1):
        self.visibleLayer = miniBatch
        for _ in range(nCDSteps):
            visibleOut, hiddenOut = self.gibbsSample()
        #visibleOut = self.rollBernoulliProbabilities(visibleOut)
        sampleError = miniBatch - visibleOut
        meanSquaredError = (sampleError * sampleError).mean()
        return meanSquaredError

    def computeFreeEnergy(self, miniBatch=None):
        if miniBatch is not None:
            self.visibleLayer = miniBatch
        internalFE = -self.visibleLayer @ self.visibleBias
        externalConditionalE = self.hiddenBias + self.visibleLayer@self.weights
        externalFE = -np.log(1. + np.exp(externalConditionalE)).sum(axis=1)
        return internalFE + externalFE

    def computeMeanFreeEnergy(self, miniBatch=None):
        return self.computeFreeEnergy(miniBatch).mean()


    #
    #miscellaneous methods
    #

    def copy(self):
        copyRBM = RestrictedBoltzmannMachine(np.copy(self.visibleLayer),
                                             np.copy(self.hiddenLayer),
                                             temperature=self.temperature,
                                             rngState=self.rng.get_state())
        copyRBM.visibleBias = np.copy(self.visibleBias)
        copyRBM.hiddenBias = np.copy(self.hiddenBias)
        copyRBM.weights = np.copy(self.weights)
        copyRBM.visibleStep = np.copy(self.visibleStep)
        copyRBM.hiddenStep = np.copy(self.hiddenStep)
        copyRBM.weightStep = np.copy(self.weightStep)
        return copyRBM

    def storeHiddenActivationsOnMiniBatch(self, miniBatch, hiddenUnits=None):
        self.visibleLayer = miniBatch
        self.hiddenConditionalProbabilities()
        return np.copy(self.hiddenLayer) if hiddenUnits is None \
          else np.copy(self.hiddenLayer[..., hiddenUnits])

    def setRngSeed(self, rngSeed):
        self.rng.seed(rngSeed)

    def __len__(self):
        return self.visibleLayer.shape[-1], self.hiddenLayer.shape[-1]

def logistic(x):
    return 1. / (1. + np.exp(-x))
