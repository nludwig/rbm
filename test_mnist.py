import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.random import RandomState
from rbm import RestrictedBoltzmannMachine
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

def getMiniBatchByLabel(dataByLabel, size, rng, shuffle=True):
    assert size % len(dataByLabel) == 0
    perLabel = size // len(dataByLabel)
    vectorLength = dataByLabel[0].shape[-1]
    miniBatch = np.array([])
    for data in dataByLabel:
        miniBatch = np.append(miniBatch, getMiniBatch(data, perLabel, rng))
    miniBatch = miniBatch.reshape((len(miniBatch)//vectorLength, vectorLength))
    if shuffle:
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
    #gradientWalker = 'sgd'
    gradientWalker = 'adam'
    iterations, miniBatchSize = int(5e6), 10
    internalRngSeed, externalRngSeed = 1337, 1234
    rng = RandomState(seed=externalRngSeed)
    plotNumber, plotStride = 5, 1
    trainingReconstructionErrorOutputStride = 10
    trainingOutputStride = iterations // 5
    #l1Coefficient = 1e-5
    l1Coefficient = None
    l2Coefficient = 1e-4
    parameterFileNameIn, parameterFileNameOut = f'mnistRBM-{gradientWalker}-1000000step.para', f'mnistRBM-{gradientWalker}-{iterations+1000000}step.para'
    runTraining = True
    verbose = False
    mnistReconProbPlotFilePrefix = f'mnistReconProb-{gradientWalker}-{iterations}steps-'
    mnistReconPlotFilePrefix = f'mnistRecon-{gradientWalker}-{iterations}steps-'
    parameterHistogramFilePrefix = f'paraHistogram-{gradientWalker}-{iterations}steps-'
    gradientHistogramFilePrefix = f'gradHistogram-{gradientWalker}-{iterations}steps-'
    numReceptiveFields, receptiveFieldFilePrefix = 9, f'receptiveField-{gradientWalker}-{iterations}steps-'
    hiddenUnitActivationsSubset = rng.randint(numberHiddenUnits, size=numberHiddenUnits//10)
    hiddenUnitActivationFilePrefix = f'hiddenUnitActivation-{gradientWalker}-{iterations}steps-'
    feFileName = f'fe-{gradientWalker}-{iterations}steps.pdf'
    feRatioFileName = f'feRatio-{gradientWalker}-{iterations}steps.pdf'
    if gradientWalker == 'sgd':
        learningRate = 1e-4
    elif gradientWalker == 'adam':
        learningRate = 1e-4
        adams = dict(zip(['visible', 'hidden', 'weights'],
                         [Adam(stepSize=learningRate) for _ in range(3)]))
    else:
        exit(1)

    #setup RBM
    visibleProportionOn = np.sum([images.sum(axis=0) for images in trainImagesByLabel], axis=0) / trainImages.shape[0]
    #visibleProportionOn = None
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
                                                              l1Coefficient=l1Coefficient,
                                                              l2Coefficient=l2Coefficient,
                                                              verbose=verbose)
    elif gradientWalker == 'adam':
        updateParameters = lambda miniBatch, \
                                  miniFantasyBatch: \
                                      rbm.updateParametersAdam(miniBatch,
                                                               miniFantasyBatch,
                                                               adams,
                                                               nCDSteps=nCDSteps,
                                                               l1Coefficient=l1Coefficient,
                                                               l2Coefficient=l2Coefficient,
                                                               verbose=verbose)
    else:
        exit(1)

    #build dict for parameter histogram output
    weightParameterTypes = {'Visible': rbm.visibleBias,
                            'Hidden': rbm.hiddenBias,
                            'Weights': rbm.weights}
    weightHistogramsByParameterType = {'Visible': [],
                                      'Hidden': [],
                                      'Weights': []}
    gradientParameterTypes = {'Visible': rbm.visibleStep,
                              'Hidden': rbm.hiddenStep,
                              'Weights': rbm.weightStep}
    gradientHistogramsByParameterType = {'Visible': [],
                                         'Hidden': [],
                                         'Weights': []}
    historicalRBMs = []
    hiddenUnitActivations = []
    historicalFEs = []
    trainSamplesForFE = getMiniBatchByLabel(trainImagesByLabel, miniBatchSize*10, rng)
    testSamplesForFE = getMiniBatchByLabel(testImagesByLabel, miniBatchSize*10, rng)
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
            if (i+1) % trainingOutputStride == 0:
                for parameterType in weightParameterTypes:
                    xs, ys, _ = diagnostics.computeHistogramArray(weightParameterTypes[parameterType].flatten())
                    weightHistogramsByParameterType[parameterType].append((i, xs, ys))
                    xs, ys, _ = diagnostics.computeHistogramArray(gradientParameterTypes[parameterType].flatten())
                    gradientHistogramsByParameterType[parameterType].append((i, xs, ys))
                historicalRBMs.append((i, rbm.copy()))
                hiddenUnitActivations.append((i,
                    rbm.storeHiddenActivationsOnMiniBatch(miniBatch,
                                                          hiddenUnits=hiddenUnitActivationsSubset)))
                historicalFEs.append((i,
                                      rbm.computeMeanFreeEnergy(trainSamplesForFE),
                                      rbm.computeMeanFreeEnergy(testSamplesForFE)))


        rbm.hiddenConditionalProbabilities()

        loopEndTime = time.time()

        if parameterFileNameOut is not None:
            with open(parameterFileNameOut, 'w') as parameterFile:
                rbm.dumpParameterFile(parameterFile)

    outputStartTime = time.time()
    #plot reconstruction series
    reconstructionPlots = []
    reconstructionProbPlots = []
    visibleStarts = getMiniBatchByLabel(testImagesByLabel, 10, rng)
    reconstructionPlots.append([visible] for visible in visibleStarts)
    reconstructionProbPlots.append([visible] for visible in visibleStarts)
    for i, visible in enumerate(visibleStarts):
        rbm.visibleLayer = visible
        for _ in range(plotNumber-1):
            for _ in range(plotStride):
                visible, _ = rbm.gibbsSample(hiddenUnitsStochastic=False)
            reconstructionProbPlots.append(visible)
            reconstructionPlots.append(rbm.rollBernoulliProbabilities(visible))
        plotMNISTSeries(reconstructionProbPlots, fileName=''.join((mnistReconProbPlotFilePrefix, f'{i}.pdf')))
        plotMNISTSeries(reconstructionPlots, fileName=''.join((mnistReconPlotFilePrefix, f'{i}.pdf')))

    #plot parameter histograms
    print('#step\tparaLo\tparaHi\tgradLo\tgradHi\tratioLo\tratioHi')
    for parameterType in weightHistogramsByParameterType:
        print(parameterType)
        for i, (step, xs, ys) in enumerate(weightHistogramsByParameterType[parameterType]):
            print('{}\t{:.3f}\t{:.3f}'.format(step, xs.min(), xs.max()), end='\t')
            _, gradxs, gradys = gradientHistogramsByParameterType[parameterType][i]
            print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(gradxs.min(), gradxs.max(),
                                        gradxs.min()/xs.min(), gradxs.max()/xs.max()))

            #parameterHistogramFileName = ''.join((parameterHistogramFilePrefix,
            #                                      parameterType,
            #                                      f'{step}.pdf'))
            #diagnostics.plotHistogramArray(xs, ys, title=parameterType+f' {step}',
            #                               fileName=parameterHistogramFileName)

            #gradientHistogramFileName = ''.join((gradientHistogramFilePrefix,
            #                                     parameterType,
            #                                     f'{step}.pdf'))
            #diagnostics.plotHistogramArray(gradxs, gradys, title=parameterType+f' grad {step}',
            #                               fileName=gradientHistogramFileName)

        parameterHistogramTSFileName = ''.join((parameterHistogramFilePrefix,
                                                parameterType,
                                                f'TimeSeries.pdf'))
        diagnostics.plotHistogramArraySeries(
            [weightHistogram[1] for weightHistogram in weightHistogramsByParameterType[parameterType]],
            [weightHistogram[2] for weightHistogram in weightHistogramsByParameterType[parameterType]],
            title=parameterType+' time series',
            fileName=parameterHistogramTSFileName)
        gradientHistogramTSFileName = ''.join((gradientHistogramFilePrefix,
                                               parameterType,
                                               f'TimeSeries.pdf'))
        diagnostics.plotHistogramArraySeries(
            [gradientHistogram[1] for gradientHistogram in gradientHistogramsByParameterType[parameterType]],
            [gradientHistogram[2] for gradientHistogram in gradientHistogramsByParameterType[parameterType]],
            title=parameterType+' grad time series',
            fileName=gradientHistogramTSFileName)
    
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

    #plot FE vs time
    t = [fe[0] for fe in historicalFEs]
    trainFE = [fe[1] for fe in historicalFEs]
    testFE = [fe[2] for fe in historicalFEs]
    diagnostics.plotTrainingTestAverageFEVsTime(t, trainFE, testFE, fileName=feFileName)
    diagnostics.plotTrainingTestAverageFEVsTime(t, trainFE/testFE, None, fileName=feRatioFileName)

    outputEndTime = time.time()
    print(f'setup time {setupEndTime-setupStartTime}s')
    if runTraining is True:
        print(f'training loop time {loopEndTime-loopStartTime}s')
    print(f'output time {outputEndTime-outputStartTime}s')

if __name__ == '__main__':
    main()
