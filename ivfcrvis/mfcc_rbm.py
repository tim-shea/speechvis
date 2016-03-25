import numpy
import matplotlib.pyplot as pyplot
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from ivfcrvis import recording as rec
from ivfcrvis import segment as seg
from ivfcrvis.mfcc_analysis import speaker_mfccs
import scipy.spatial as spatial


def scale_mfccs(mfccs):
    lower = numpy.reshape(numpy.min(mfccs, axis=1), (mfccs.shape[0], 1))
    upper = numpy.reshape(numpy.max(mfccs, axis=1), (mfccs.shape[0], 1))
    return (mfccs - lower) / (upper - lower)


def train_rbm(recording):
    index, _, _ = recording.filter_speaker('CHN')
    starts, ends, mfccs = speaker_mfccs(recording, 'CHN', )
    data = scale_mfccs(mfccs)
    layer1 = BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=20)
    layer2 = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=20)
    layer3 = BernoulliRBM(n_components=16, learning_rate=0.01, n_iter=20)
    pipeline = Pipeline(steps=[('l2', layer2), ('l3', layer3)])
    pipeline.fit(data)
    x = pipeline.transform(data)

    fig, ax = pyplot.subplots()
    for i in range(16):
        pyplot.subplot(4, 4, i + 1)
        pyplot.plot(x[:,i])
        pyplot.title('Unit {0}'.format(i))
