import numpy as np
from matplotlib import pyplot as plt
from numpy.random import RandomState
from layers import RestrictedBoltzmannMachine

def loadMNIST():
    from tensorflow.keras.datasets import mnist
    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()
    newTrainShape = (trainImages.shape[0], trainImages.shape[1]*trainImages.shape[2])
    newTestShape = (testImages.shape[0], testImages.shape[1]*testImages.shape[2])
    trainImages = np.reshape(trainImages, newTrainShape)
    testImages = np.reshape(testImages, newTestShape)
    return trainImages, testImages

def binarize(data):
    maxValue = data.max()
    return (data > maxValue//2).astype(np.float_)

def getMiniBatch(data, size, rng):
    indices = rng.randint(data.shape[0], size=size)
    return data[indices]

def plotMNIST(image):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.show()

def plotMNISTSeries(images):
    fig, axs = plt.subplots(1, len(images))
    for i, ax in enumerate(axs):
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
    plt.show()

def main():
    #load data
    trainImages, testImages = loadMNIST()
    images = binarize(np.append(trainImages, testImages, axis=0))

    #parameters
    numberVisibleUnits = images.shape[1]
    numberHiddenUnits = int(numberVisibleUnits * 2./3.)
    temperature, sigma, nCDSteps = 1., 0.01, 1
    iterations, miniBatchSize, learningRate = 100, 100, 0.01
    internalRngSeed, externalRngSeed = 1337, 1234
    plotStartIndex, plotNumber, plotSkip = 100, 5, 100
    trainingOutputSkip = 10
    parameterFileNameIn, parameterFileNameOut = None, 'parameterFile.txt'
    rng = RandomState(seed=externalRngSeed)

    #setup RBM
    visibleLayer = np.zeros(numberVisibleUnits)
    hiddenLayer = np.zeros(numberHiddenUnits)
    if parameterFileNameIn is not None:
        with open(parameterFileNameIn, 'r') as parameterFile:
            rbm = RestrictedBoltzmannMachine(visibleLayer, hiddenLayer,
                temperature=temperature, sigma=sigma,
                visibleProportionOn=images.mean(axis=0),
                parameterFile=parameterFile, rngSeed=internalRngSeed)
    else:
        rbm = RestrictedBoltzmannMachine(visibleLayer, hiddenLayer,
            temperature=temperature, sigma=sigma,
            visibleProportionOn=images.mean(axis=0), rngSeed=internalRngSeed)

    #build fantasy batch
    miniFantasyBatch = np.copy(getMiniBatch(images, miniBatchSize, rng))

    for i in range(iterations):
        miniBatch = getMiniBatch(images, miniBatchSize, rng)
        miniFantasyBatch = rbm.updateParametersSGD(miniBatch, miniFantasyBatch, learningRate, nCDSteps=nCDSteps)
        if (i+1) % trainingOutputSkip == 0:
            print(f'step {i+1} / {iterations}', flush=True)

    with open(parameterFileNameOut, 'w') as parameterFile:
        rbm.dumpParameterFile(parameterFile)

    plots = []
    visible = images[plotStartIndex]
    plots.append(visible)
    rbm.visibleLayer = visible
    for i in range(plotNumber-1):
        for j in range(plotSkip):
            visible, _ = rbm.gibbsSample()
        plots.append(visible)
    plotMNISTSeries(plots)

if __name__ == '__main__':
    main()
