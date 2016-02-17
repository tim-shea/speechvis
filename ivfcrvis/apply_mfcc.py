import numpy
from ivfcrvis import split_vocalizations
from features import mfcc
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
from matplotlib import pyplot


ivfcr_root = 'D:/ivfcr/'
subject_id = 'e20131030_125347_009146'


def get_vocalization(category, vocalization_no):
    return wavfile.read(ivfcr_root + subject_id + '/{0}/{1}.wav'.format(category, vocalization_no))


def apply_mfcc(sample_rate, vocalization):
    return mfcc(vocalization[:int(0.6 * sample_rate)], sample_rate)


def show_mfcc():
    starts, ends, categories = split_vocalizations.get_vocalization_labels()
    index = numpy.where(numpy.array(categories) == 'CHN')[0]
    pyplot.figure()
    for i,j in zip(index[:4], range(1, 5)):
        sample_rate, vocalization = get_vocalization('CHN', i)
        mfccs = apply_mfcc(sample_rate, vocalization)
        print(numpy.shape(mfccs))
        pyplot.subplot(2, 2, j)
        pyplot.title('Vocalization {0} MFCC'.format(i))
        pyplot.imshow(mfccs.transpose(), extent=[0, 600, 0, 13], aspect='auto', interpolation='nearest')
        pyplot.xlabel('Time (ms)')
        pyplot.ylabel('Coefficient')
        pyplot.colorbar()


def plot_mfcc_feature(coord):
    starts, ends, categories = split_vocalizations.get_vocalization_labels()
    index = numpy.where(numpy.array(categories) == 'CHN')[0]
    x = []
    for i in index:
        sample_rate, vocalization = get_vocalization('CHN', i)
        mfccs = apply_mfcc(sample_rate, vocalization)
        x.append(mfccs[coord[0], coord[1]])
    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.title('MFCC Feature ({0},{1}) Over Time'.format(coord[0], coord[1]))
    pyplot.plot(starts[index], x)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('MFCC')
    pyplot.subplot(2, 1, 2)
    pyplot.hist(x, bins=50, normed=True)
    pyplot.xlabel('MFCC')
    pyplot.ylabel('Frequency')


def plot_mfcc_2feature(xCoord, yCoord):
    starts, ends, categories = split_vocalizations.get_vocalization_labels()
    index = numpy.where(numpy.array(categories) == 'CHN')[0]
    x = []
    y = []
    for i in index:
        sample_rate, vocalization = get_vocalization('CHN', i)
        mfccs = apply_mfcc(sample_rate, vocalization)
        x.append(mfccs[xCoord[0], xCoord[1]])
        y.append(mfccs[yCoord[0], yCoord[1]])
    pyplot.figure()
    pyplot.plot(x, y)


def get_mfcc_distances(category):
    starts, ends, categories = split_vocalizations.get_vocalization_labels()
    index = numpy.where(numpy.array(categories) == category)[0]
    previous = numpy.zeros(59 * 13)
    x = numpy.zeros(len(index))
    for i,j in zip(index, range(len(index))):
        sample_rate, vocalization = get_vocalization(category, i)
        mfccs = numpy.reshape(apply_mfcc(sample_rate, vocalization), 59 * 13)
        x[j] = euclidean(previous, mfccs)
        previous = mfccs
    return starts[index], ends[index], x
