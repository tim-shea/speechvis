import os
import numpy
from xml.etree import ElementTree
from scipy.io import wavfile
from matplotlib import pyplot


ivfcr_root = 'D:/ivfcr/'
subject_id = 'e20131030_125347_009146'


def parse_time(formatted):
    # TODO: This should not require Pacific timezone, lookup lena format spec
    if formatted.startswith('PT') and formatted.endswith('S'):
        return float(formatted[2:-1])


def get_vocalization_labels():
    starts = []
    ends = []
    categories = []
    tree = ElementTree.parse(ivfcr_root + subject_id + '.its')
    root = tree.getroot()
    for segment in root.iter('Segment'):
        categories.append(segment.attrib['spkr'])
        starts.append(parse_time(segment.attrib['startTime']))
        ends.append(parse_time(segment.attrib['endTime']))
    return numpy.array(starts), numpy.array(ends), categories


def filter_vocalization_category(starts, ends, labels, category):
    index = numpy.array(labels) == category
    return starts[index], ends[index]


def get_recording():
    return wavfile.read(ivfcr_root + subject_id + '.wav')


def split_vocalizations(starts, ends, categories, sample_rate, recording):
    sequence = 0
    if not os.path.exists(ivfcr_root + '/' + subject_id):
        os.makedirs(ivfcr_root + '/' + subject_id)
    for category in set(categories):
        if not os.path.exists(ivfcr_root + '/' + subject_id + '/' + category):
            os.makedirs(ivfcr_root + '/' + subject_id + '/' + category)
    for start, end, category in zip(starts, ends, categories):
        data = recording[int(start * sample_rate):int(end * sample_rate)]
        wavfile.write(ivfcr_root + subject_id + '/{0}/{1}.wav'.format(category, sequence), sample_rate, data)
        sequence = sequence + 1


def plot_durations(starts, ends):
    pyplot.figure()
    durations = ends - starts
    pyplot.subplot(2, 1, 1)
    pyplot.plot(starts + durations / 2, durations)
    pyplot.subplot(2, 1, 2)
    #pyplot.hist(durations, bins=numpy.logspace(0, 4))
    pyplot.plot(numpy.sort(durations), numpy.linspace(0, 1, numpy.size(durations)))


def plot_intervals(starts, ends):
    pyplot.figure()
    intervals = starts[1:] - ends[:-1]
    pyplot.subplot(2, 1, 1)
    pyplot.plot(starts[1:], intervals)
    pyplot.subplot(2, 1, 2)
    #pyplot.hist(intervals, bins=numpy.logspace(0, 4))
    pyplot.plot(numpy.sort(intervals), numpy.linspace(0, 1, numpy.size(intervals)))


if __name__ == "__main__":
    starts, ends, categories = get_vocalization_labels()
    sample_rate, recording = get_recording()
    split_vocalizations(starts, ends, categories, sample_rate, recording)
