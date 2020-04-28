import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.random import RandomState
from layers import RestrictedBoltzmannMachine
from adam import Adam
import diagnostics

def loadMNIST():
    #load
    from tensorflow.keras.datasets import mnist
    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()
    #flatten
    newTrainShape = (trainImages.shape[0], trainImages.shape[1]*trainImages.shape[2])
    newTestShape = (testImages.shape[0], testImages.shape[1]*testImages.shape[2])
    trainImages = trainImages.reshape(newTrainShape)
    testImages = testImages.reshape(newTestShape)
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

def plotMNIST(image, mnistRows=28, mnistColumns=28):
    diagnostics.plotImage(rows=mnistRows,
                          columns=mnistColumns)

def plotMNISTSeries(images, mnistRows=28, mnistColumns=28, norm=None, fileName=None):
    diagnostics.plotImageSeries(images,
                                rows=mnistRows,
                                columns=mnistColumns,
                                norm=norm,
                                fileName=fileName)

def plotReceptiveFields(rbm, hiddenUnitIndices, fileName=None):
    norm = matplotlib.colors.Normalize(vmin=rbm.weights[:, hiddenUnitIndices].min(),
                                       vmax=rbm.weights[:, hiddenUnitIndices].max())
    plotMNISTSeries(rbm.weights[:, hiddenUnitIndices].T, norm=norm, fileName=fileName)

def main():
    setupStartTime = time.time()
    #load data
    trainImages, trainLabels, testImages, testLabels = loadMNIST()
    trainImages = binarize(trainImages)
    testImages = binarize(testImages)
    trainImagesByLabel, testImagesByLabel = \
            sortMNISTByLabel(trainImages, trainLabels, testImages, testLabels)

    #parameters
    numberVisibleUnits = trainImages.shape[-1]
    numberHiddenUnits = 200
    #numberHiddenUnits = int(numberVisibleUnits * 2./3.)
    temperature, nCDSteps = 1., 1
    #sigma = 0.01
    sigma = 2. / np.sqrt(numberVisibleUnits + numberHiddenUnits)
    gradientWalker = 'sgd'
    #gradientWalker = 'adam'
    iterations, miniBatchSize = int(1e5), 10
    internalRngSeed, externalRngSeed = 1337, 1234
    rng = RandomState(seed=externalRngSeed)
    plotStartIndex, plotNumber, plotStride = 100, 5, 1
    trainingReconstructionErrorOutputStride = 10
    trainingHistogramOutputStride = iterations // 5
    l2Coefficient = 1e-2
    parameterFileNameIn, parameterFileNameOut = None, f'mnistRBM-{gradientWalker}-{iterations}step.para'
    runTraining = True
    verbose = False
    mnistSeriesPlotFileName = f'mnistSeries-{gradientWalker}-{iterations}steps.pdf'
    parameterHistogramFilePrefix = f'histogram-{gradientWalker}-{iterations}steps-'
    numReceptiveFields, receptiveFieldFilePrefix = 9, f'receptiveField-{gradientWalker}-{iterations}steps-'
    #hiddenUnitActivationsSubset = rng.randint(numberHiddenUnits, size=miniBatchSize)
    hiddenUnitActivationsSubset = None
    hiddenUnitActivationFilePrefix = f'hiddenUnitActivation-{gradientWalker}-{iterations}steps-'
    if gradientWalker == 'sgd':
        learningRate = 1e-4
    elif gradientWalker == 'adam':
        learningRate = 1e-6
        adams = dict(zip(['visible', 'hidden', 'weights'],
                         [Adam(stepSize=learningRate) for _ in range(3)]))
    else:
        exit(1)

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

    if gradientWalker == 'sgd':
        updateParameters = lambda miniBatch, \
                                  miniFantasyBatch: \
                                      rbm.updateParametersSGD(miniBatch,
                                                              miniFantasyBatch,
                                                              learningRate,
                                                              nCDSteps=nCDSteps,
                                                              l2Coefficient=l2Coefficient,
                                                              verbose=verbose)
    elif gradientWalker == 'adam':
        updateParameters = lambda miniBatch, \
                                  miniFantasyBatch: \
                                      rbm.updateParametersAdam(miniBatch,
                                                               miniFantasyBatch,
                                                               adams,
                                                               nCDSteps=nCDSteps,
                                                               l2Coefficient=l2Coefficient,
                                                               verbose=verbose)
    else:
        exit(1)

    #build dict for parameter histogram output
    parameterTypes = {'Visible': rbm.visibleBias,
                      'Hidden': rbm.hiddenBias,
                      'Weights': rbm.weights}
    histogramsByParameterType = {'Visible': [],
                                 'Hidden': [],
                                 'Weights': []}
    historicalRBMs = []
    hiddenUnitActivations = []
    setupEndTime = time.time()

    if runTraining is True:
        loopStartTime = time.time()
        #build fantasy batch
        miniFantasyBatch = np.copy(getMiniBatchByLabel(trainImagesByLabel, miniBatchSize, rng))
        for i in range(iterations):
            miniBatch = getMiniBatchByLabel(trainImagesByLabel, miniBatchSize, rng)
            miniFantasyBatch = updateParameters(miniBatch, miniFantasyBatch)
            if (i+1) % trainingReconstructionErrorOutputStride == 0:
                print(i, rbm.computeReconstructionError(getMiniBatchByLabel(testImagesByLabel, miniBatchSize, rng)))
            if (i+1) % trainingHistogramOutputStride == 0:
                for parameterType in parameterTypes:
                    xs, ys, _ = diagnostics.computeHistogramArray(parameterTypes[parameterType].flatten())
                    histogramsByParameterType[parameterType].append((i, xs, ys))
                historicalRBMs.append((i, rbm.copy()))
                hiddenUnitActivations.append((i,
                    rbm.storeHiddenActivationsOnMiniBatch(miniBatch,
                                                          hiddenUnits=hiddenUnitActivationsSubset)))


        rbm.hiddenConditionalProbabilities()

        loopEndTime = time.time()

        if parameterFileNameOut is not None:
            with open(parameterFileNameOut, 'w') as parameterFile:
                rbm.dumpParameterFile(parameterFile)

    outputStartTime = time.time()
    #plot reconstruction series
    plots = []
    visible = testImages[plotStartIndex]
    plots.append(visible)
    rbm.visibleLayer = np.copy(visible)
    for i in range(plotNumber-1):
        for j in range(plotStride):
            visible, _ = rbm.gibbsSample(hiddenUnitsStochastic=False)
        #plots.append(visible)
        plots.append(rbm.rollBernoulliProbabilities(visible))
    plotMNISTSeries(plots, fileName=mnistSeriesPlotFileName)

    #plot parameter histograms
    for parameterType in histogramsByParameterType:
        print(parameterType)
        for i, xs, ys in histogramsByParameterType[parameterType]:
            print(i, xs.min(), xs.max())
            parameterHistogramFileName = ''.join((parameterHistogramFilePrefix,
                                                  parameterType,
                                                  f'{i}.pdf'))
            diagnostics.plotHistogramArray(xs, ys, title=parameterType+f' {i}',
                                           fileName=parameterHistogramFileName)
    
    #plot receptive fields
    hiddenUnitIndices = rng.randint(rbm.hiddenBias.shape[0], size=numReceptiveFields)
    for i, historicalRBM in historicalRBMs:
        receptiveFieldFileName = ''.join((receptiveFieldFilePrefix,
                                          f'{i}.pdf'))
        plotReceptiveFields(historicalRBM, hiddenUnitIndices, fileName=receptiveFieldFileName)

    #plot hidden unit activations
    for i, hiddenUnitActivation in hiddenUnitActivations:
        hiddenUnitActivationFileName = ''.join((hiddenUnitActivationFilePrefix,
                                                f'{i}.pdf'))
        diagnostics.plotHiddenActivationsOnMiniBatch(hiddenUnitActivation,
                                                     fileName=hiddenUnitActivationFileName)
    outputEndTime = time.time()
    print(f'setup time {setupEndTime-setupStartTime}s')
    if runTraining is True:
        print(f'training loop time {loopEndTime-loopStartTime}s')
    print(f'output time {outputEndTime-outputStartTime}s')

if __name__ == '__main__':
    main()
