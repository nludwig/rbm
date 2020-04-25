import time
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import RandomState
from layers import RestrictedBoltzmannMachine
from adam import Adam

def loadMNIST():
    #load
    from tensorflow.keras.datasets import mnist
    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()
    #flatten
    newTrainShape = (trainImages.shape[0], trainImages.shape[1]*trainImages.shape[2])
    newTestShape = (testImages.shape[0], testImages.shape[1]*testImages.shape[2])
    trainImages = np.reshape(trainImages, newTrainShape)
    testImages = np.reshape(testImages, newTestShape)
    return trainImages, trainLabels, testImages, testLabels

def sortMNISTByLabel(trainImages, trainLabels, testImages, testLabels):
    #sort by label
    trainImagesByLabel = []
    testImagesByLabel = []
    for label in range(10):
        indices = trainLabels == label
        trainImagesByLabel.append(trainImages[indices])
        indices = testLabels == label
        testImagesByLabel.append(testImages[indices])
    return trainImagesByLabel, testImagesByLabel

def binarize(data):
    maxValue = data.max()
    return (data > maxValue//2).astype(np.float_)

def getMiniBatch(data, size, rng):
    indices = rng.randint(data.shape[0], size=size)
    return data[indices]

def getMiniBatchByLabel(dataByLabel, size, rng):
    assert size % len(dataByLabel) == 0
    perLabel = size // len(dataByLabel)
    vectorLength = dataByLabel[0].shape[-1]
    miniBatch = np.array([])
    for data in dataByLabel:
        miniBatch = np.append(miniBatch, getMiniBatch(data, perLabel, rng))
    miniBatch = miniBatch.reshape((len(miniBatch)//vectorLength, vectorLength))
    rng.shuffle(miniBatch)
    return miniBatch

def plotMNIST(image):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.show()

def plotMNISTSeries(images, fileName=None):
    fig, axs = plt.subplots(1, len(images))
    for i, ax in enumerate(axs):
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
    if fileName is None:
        plt.show()
    else:
        plt.savefig(fileName)

def main():
    #load data
    trainImages, trainLabels, testImages, testLabels = loadMNIST()
    trainImages = binarize(trainImages)
    testImages = binarize(testImages)
    trainImagesByLabel, testImagesByLabel = \
            sortMNISTByLabel(trainImages, trainLabels, testImages, testLabels)

    #parameters
    numberVisibleUnits = trainImages.shape[-1]
    numberHiddenUnits = int(numberVisibleUnits * 2./3.)
    temperature, nCDSteps = 1., 1
    #sigma = 0.01
    sigma = 2. / np.sqrt(numberVisibleUnits + numberHiddenUnits)
    iterations, miniBatchSize, learningRate = 100, 100, 0.001
    internalRngSeed, externalRngSeed = 1337, 1234
    plotStartIndex, plotNumber, plotSkip = 100, 5, 1
    trainingOutputSkip = 10
    l2Coefficient = 1e-4
    #parameterFileNameIn, parameterFileNameOut = 'parameterFile.txt', 'parameterFile.txt'
    parameterFileNameIn, parameterFileNameOut = None, 'mnistRBM-sgd-1000step.para'
    #parameterFileNameIn, parameterFileNameOut = 'mnistRBM-sgd-1000step.para', None
    runTraining = True
    verbose = False
    plotFileName = 'mnistRBM-sgd-1000step-3.pdf'
    rng = RandomState(seed=externalRngSeed)
    adams = dict(zip(['visible', 'hidden', 'weights'],
                     [Adam(stepSize=learningRate) for _ in range(3)]))

    #setup RBM
    visibleProportionOn = np.sum([images.sum(axis=0) for images in trainImagesByLabel], axis=0) / trainImages.shape[0]
    visibleLayer = np.zeros(numberVisibleUnits)
    hiddenLayer = np.zeros(numberHiddenUnits)
    if parameterFileNameIn is not None:
        with open(parameterFileNameIn, 'r') as parameterFile:
            rbm = RestrictedBoltzmannMachine(visibleLayer, hiddenLayer,
                temperature=temperature, sigma=sigma,
                visibleProportionOn=visibleProportionOn,
                parameterFile=parameterFile, rngSeed=internalRngSeed)
    else:
        rbm = RestrictedBoltzmannMachine(visibleLayer, hiddenLayer,
            temperature=temperature, sigma=sigma,
            visibleProportionOn=visibleProportionOn, rngSeed=internalRngSeed)

    if runTraining is True:
        loopStartTime = time.time()
        #build fantasy batch
        miniFantasyBatch = np.copy(getMiniBatchByLabel(trainImagesByLabel, miniBatchSize, rng))
        for i in range(iterations):
            miniBatch = getMiniBatchByLabel(trainImagesByLabel, miniBatchSize, rng)
            miniFantasyBatch = rbm.updateParametersSGD(miniBatch, miniFantasyBatch, learningRate, nCDSteps=nCDSteps, l2Coefficient=l2Coefficient, verbose=verbose)
            #miniFantasyBatch = rbm.updateParametersAdam(miniBatch, miniFantasyBatch, adams, nCDSteps=nCDSteps, l2Coefficient=l2Coefficient, verbose=verbose)
            print(rbm.computeReconstructionError(getMiniBatchByLabel(testImagesByLabel, miniBatchSize, rng)))
            #if (i+1) % trainingOutputSkip == 0:
            #    print(f'step {i+1} / {iterations}', flush=True)
        loopEndTime = time.time()
        print(f'training loop time {loopEndTime-loopStartTime}')

        if parameterFileNameOut is not None:
            with open(parameterFileNameOut, 'w') as parameterFile:
                rbm.dumpParameterFile(parameterFile)

    plots = []
    visible = testImages[plotStartIndex]
    plots.append(visible)
    rbm.visibleLayer = np.copy(visible)
    for i in range(plotNumber-1):
        for j in range(plotSkip):
            visible, _ = rbm.gibbsSample(hiddenUnitsStochastic=False)
        #plots.append(visible)
        plots.append(rbm.rollBernoulliProbabilities(visible))
    plotMNISTSeries(plots, plotFileName)

if __name__ == '__main__':
    main()
