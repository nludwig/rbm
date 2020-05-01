import numpy as np

class Adam:
    def __init__(self, stepSize=None, firstMomentRate=0.9, secondMomentRate=0.999, denominatorEpsilon=1e-8):
        if stepSize is None:
            self.stepSize = infiniteGenerator(0.001)
        elif type(stepSize) == int or type(stepSize) == float:
            self.stepSize = infiniteGenerator(stepSize)
        elif type(stepSize) == type(infiniteGenerator(0)):
            self.stepSize = stepSize
        else:
            print(f'stepSize must be type generator, int, or float, not {type(stepSize)}. exiting')
            exit(1)
        
        self.firstMomentRate = firstMomentRate
        self.secondMomentRate = secondMomentRate
        self.denominatorEpsilon = denominatorEpsilon

        self.t = 1

        self.firstMoment = 0.
        self.secondMoment = 0.
        self.correctedFirstMoment = 0.
        self.correctedSecondMomemnt = 0.

    def updateMoments(self, gradient):
        self.firstMoment = self.firstMomentRate * self.firstMoment \
                           + (1. - self.firstMomentRate) * gradient
        self.secondMoment = self.secondMomentRate * self.secondMoment \
                            + (1. - self.secondMomentRate) * gradient * gradient

    def computeAdamStepSlow(self, gradient):
        self.updateMoments(gradient)
        self.correctedFirstMoment = self.firstMoment / (1. - self.firstMomentRate**self.t)
        self.correctedSecondMoment = self.secondMoment / (1. - self.secondMomentRate**self.t)
        selt.t += 1
        return next(self.stepSize) * self.correctedFirstMoment \
                / (np.sqrt(self.correctedSecondMoment) + self.denominatorEpsilon)

    def computeAdamStep(self, gradient):
        self.updateMoments(gradient)
        scaledStepSize = next(self.stepSize) * np.sqrt(1. - self.secondMomentRate**self.t) \
                            / (1. - self.firstMomentRate**self.t)
        self.t += 1
        return scaledStepSize * self.firstMoment \
                / (np.sqrt(self.secondMoment) + self.denominatorEpsilon)

def infiniteGenerator(number):
    while True:
        yield number
