import numpy as np
from matplotlib import pyplot as plt

#
#plotting
#

def plotImage(image, rows, columnns):
    plt.imshow(image.reshape(rows, columns), cmap='gray')
    plt.show()
    plt.clf()

def plotImageSeries(images, rows, columns, norm=None, fileName=None):
    rootNumberImages = np.sqrt(len(images))
    if int(rootNumberImages) == rootNumberImages:
        fig, axs = plt.subplots(nrows=int(rootNumberImages),
                                ncols=int(rootNumberImages),
                                sharex=True,
                                sharey=True)
    else:
        fig, axs = plt.subplots(nrows=1,
                                ncols=len(images),
                                sharey=True)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i].reshape(rows, columns), cmap='gray', norm=norm)
    if fileName is None:
        plt.show()
    else:
        plt.savefig(fileName)
    plt.clf()

def plotHistogramArray(binCenters, histogram, title='', fileName=None):
    plt.plot(binCenters, histogram)
    plt.title(title)
    if fileName is None:
        plt.show()
    else:
        plt.savefig(fileName)
    plt.clf()

def plotHistogramArraySeries(binCenterss, histograms, title='', fileName=None):
    for i, binCenters in enumerate(binCenterss):
        plt.plot(binCenters, histograms[i], label=f'{i}')
    plt.title(title)
    plt.legend()
    if fileName is None:
        plt.show()
    else:
        plt.savefig(fileName)
    plt.clf()

def plotHiddenActivationsOnMiniBatch(activations, fileName=None):
    plt.imshow(activations, cmap='gray')
    if fileName is None:
        plt.show()
    else:
        plt.savefig(fileName)
    plt.clf()


#
#statistics & data crunching
#

def computeHistogramList(samples, delta=1.):
    low, high = min(samples), max(samples)
    low = int(low // delta) * delta
    high = int(high // delta + 2) * delta
    histoLength = int((high - low) / delta)
    #print(low, high, histoLength)
    xs = [i*delta + low for i in range(histoLength)]
    ys = [0 for _ in range(histoLength)]
    for sample in samples:
        i = bisect_left(xs, sample)
        #print(i, histoLength, sample, low, high)
        ys[i] += 1
    #change bin x from low part to mid of bin
    xs = [x + 0.5*delta for x in xs]
    return xs, ys

def computeHistogramArray(samples):
    histogram, binEdges = np.histogram(samples, bins='auto')
    binCenters = (binEdges[:-1] + binEdges[1:]) / 2.
    return binCenters, histogram, binEdges
