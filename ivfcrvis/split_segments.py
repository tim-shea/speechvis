import os
import numpy
from xml.etree import ElementTree
from scipy.io import wavfile
from matplotlib import pyplot


default_root = 'D:/ivfcr'
sample_recording_id = 'e20131030_125347_009146'


def parse_time(formatted):
    # TODO: This should not require Pacific timezone, lookup lena format spec
    if formatted.startswith('PT') and formatted.endswith('S'):
        return float(formatted[2:-1])


def read_segment_labels(root, recording_id):
    starts = []
    ends = []
    speakers = []
    tree = ElementTree.parse('{0}/{1}.its'.format(root, recording_id))
    root = tree.getroot()
    for segment in root.iter('Segment'):
        speakers.append(segment.attrib['spkr'])
        starts.append(parse_time(segment.attrib['startTime']))
        ends.append(parse_time(segment.attrib['endTime']))
    return numpy.array(starts), numpy.array(ends), speakers


def filter_segments_by_speaker(starts, ends, labels, speaker):
    index = numpy.array(labels) == speaker
    return starts[index], ends[index]


def read_recording(root, id):
    return wavfile.read(root + id + '.wav')


def read_segment(root, id, category, number):
    return wavfile.read('{0}/{1}/{2}/{3}.wav'.format(root, id, category, number))


def split_segments(root, id, starts, ends, speakers, sample_rate, recording):
    sequence = 0
    if not os.path.exists(root + '/' + id):
        os.makedirs(root + '/' + id)
    for speaker in set(speakers):
        if not os.path.exists(root + '/' + id + '/' + speaker):
            os.makedirs(root + '/' + id + '/' + speaker)
    for start, end, speaker in zip(starts, ends, speakers):
        data = recording[int(start * sample_rate):int(end * sample_rate)]
        wavfile.write(root + id + '/{0}/{1}.wav'.format(speaker, sequence), sample_rate, data)
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
