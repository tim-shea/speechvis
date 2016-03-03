import numpy
import matplotlib.pyplot as pyplot
from sklearn.neural_network import BernoulliRBM
from ivfcrvis import split_segments
from ivfcrvis.apply_mfcc import *
import scipy.spatial as spatial


if __name__ == "__main__":
    num_vocalizations = 2000
    starts, ends, categories = split_segments.get_vocalization_labels()
    index = numpy.where(numpy.array(categories) == 'CHN')[0]
    mfccs = numpy.ndarray((num_vocalizations, 767))
    for i in range(num_vocalizations):
        sample_rate, vocalization = get_vocalization('CHN', index[i])
        mfccs[i,:] = numpy.reshape(apply_mfcc(sample_rate, vocalization), (1, 767))
        mfccs[i,:] = mfccs[i,:] - numpy.min(mfccs[i,:])
        mfccs[i,:] = mfccs[i,:] / numpy.max(mfccs[i,:])
    rbm = BernoulliRBM(n_components=16)
    x = rbm.fit_transform(mfccs)

    fig, ax = pyplot.subplots()
    columns = numpy.size(x, axis=1)
    for i in range(columns):
        pyplot.subplot(4, int(columns / 4), i + 1)
        pyplot.plot(x[:,i])
        pyplot.title('Unit {0}'.format(i))
