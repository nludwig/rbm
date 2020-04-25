import numpy as np

class Adam:
    def __init__(self, stepSize=0.001, firstMomentRate=0.9, secondMomentRate=0.999, denominatorEpsilon=1e-8):
        self.stepSize = stepSize
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
        return self.stepSize * self.correctedFirstMoment \
                / (np.sqrt(self.correctedSecondMoment) + self.denominatorEpsilon)

    def computeAdamStep(self, gradient):
        self.updateMoments(gradient)
        scaledStepSize = self.stepSize * np.sqrt(1. - self.secondMomentRate**self.t) \
                            / (1. - self.firstMomentRate**self.t)
        self.t += 1
        return scaledStepSize * self.firstMoment \
                / (self.secondMoment + self.denominatorEpsilon)
