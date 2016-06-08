import numpy
from ivfcrvis.recording import Recording
from ivfcrvis.segment import Segment
from scipy.stats import pearsonr
from matplotlib import pyplot


def speaker_frequency_banks(recording, speaker):
    """Returns the start times, end times, and Mel-frequency filter banks for every segment labeled as speaker in the
    recording."""
    index, starts, ends = recording.filter_speaker(speaker)
    fbanks = numpy.ndarray((len(starts), 312))
    for i in range(len(index)):
        fbank = Segment(recording, speaker, index[i]).frequencybank()
        fbanks[i,:] = fbank.reshape(fbank.size)
    return starts, ends, fbanks


def speaker_mfccs(recording, speaker):
    """Returns the start times, end times, and MFCCs for every segment labeled as speaker in the recording."""
    index, starts, ends = recording.filter_speaker(speaker)
    mfccs = numpy.ndarray((len(starts), 480))
    for i in range(len(index)):
        mfccs[i,:] = numpy.reshape(Segment(recording, speaker, index[i]).mfccs(), (480))
    return starts, ends, mfccs


def plot_mfcc_feature(recording, speaker, x):
    """Plot a time series and histogram of a particular feature (pixel) in the set of MFCCs for a speaker."""
    starts, ends, mfccs = speaker_mfccs(recording, speaker)
    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.title('MFCC Feature {0} Over Time'.format(x))
    pyplot.plot(starts, mfccs[:, x])
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('MFCC')
    pyplot.subplot(2, 1, 2)
    pyplot.hist(mfccs[:, x], bins=50, normed=True)
    pyplot.xlabel('MFCC')
    pyplot.ylabel('Frequency')


def plot_mfcc_steps(recording, speaker):
    """Plot the time series, histogram, and scatter against interval for the MFCC step size (distance between MFCC
    pairs) for all segments labeled as speaker in the recording."""
    starts, ends, mfccs = speaker_mfccs(recording, speaker)
    steps = numpy.linalg.norm(numpy.diff(mfccs, axis=0), axis=1)
    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(starts[1:], steps)
    pyplot.title('MFCC Steps Over Time')
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('MFCC Step Size')
    pyplot.subplot(2, 2, 3)
    pyplot.hist(steps, bins=50)
    pyplot.xlabel('MFCC Step Size')
    pyplot.ylabel('Count')
    pyplot.subplot(2, 2, 4)
    r = pearsonr(steps, numpy.log(numpy.max(starts[1:] - ends[:-1], 0.001)))
    print(r)
    pyplot.plot(steps, starts[1:] - ends[:-1], '.')
    pyplot.yscale('log')
    pyplot.xlabel('MFCC Step Size')
    pyplot.ylabel('Inter-Vocalization Interval')
