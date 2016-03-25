import numpy
import os
import math
from xml.etree import ElementTree
from scipy.io import wavfile
from matplotlib import pyplot
import seaborn


class Recording:
    default_root = 'D:/ivfcr'
    ids = ['e20131030_125347_009146',
            'e20151207_200648_010576',
            'e20151210_145625_010585',
            'e20151223_131722_010570',
            'e20160102_095210_010584',
            'e20160222_122221_010581']

    def __init__(self, root=default_root, recording_id=ids[0]):
        self.root = root
        self.recording_id = recording_id
        starts = []
        ends = []
        speakers = []
        tree = ElementTree.parse(os.path.join(self.root, '{0}.its'.format(self.recording_id)))
        root = tree.getroot()
        for segment in root.iter('Segment'):
            speakers.append(segment.attrib['spkr'])
            starts.append(parse_time(segment.attrib['startTime']))
            ends.append(parse_time(segment.attrib['endTime']))
        self.starts = numpy.array(starts)
        self.ends = numpy.array(ends)
        self.speakers = speakers

    def read_recording(self):
        return wavfile.read(os.path.join(self.root, '{0}.wav'.format(self.root, self.recording_id)))

    def split_segments(self):
        recording_dir = os.path.join(self.root, self.recording_id)
        if not os.path.exists(recording_dir):
            os.makedirs(recording_dir)
        for speaker in set(self.speakers):
            speaker_dir = os.path.join(recording_dir, speaker)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)
        sample_rate, recording = self.read_recording()
        for start, end, speaker, i in zip(self.starts, self.ends, self.speakers, range(len(self.starts))):
            segment = recording[int(start * sample_rate):int(end * sample_rate)]
            wavfile.write(os.path.join(recording_dir, speaker, '{0}.wav'.format(i)), sample_rate, segment)

    def read_segment(self, category, i):
        filename = os.path.join(self.root, self.recording_id, category, '{0}.wav'.format(i))
        return wavfile.read(filename)

    def filter_speaker(self, speaker):
        index = numpy.array(self.speakers) == speaker
        return numpy.where(index)[0], self.starts[index], self.ends[index]


def parse_time(formatted):
            # TODO: This should not require Pacific timezone, lookup lena format spec
            if formatted.startswith('PT') and formatted.endswith('S'):
                return float(formatted[2:-1])


def plot_speaker_counts(recording):
    speakers, counts = numpy.unique(recording.speakers, return_counts=True)
    pyplot.figure()
    pyplot.bar(numpy.arange(len(speakers)) + 0.1, counts)
    pyplot.title('Number of Vocalizations by Speaker')
    pyplot.xticks(numpy.arange(len(speakers)) + 0.5, speakers)
    pyplot.xlim(0, len(speakers))
    pyplot.xlabel('Speaker')
    pyplot.ylabel('Count')


def plot_durations(recording, speaker=None):
    if speaker is None:
        starts = recording.starts
        ends = recording.ends
    else:
        i, starts, ends = recording.filter_speaker(speaker)
    durations = ends - starts
    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(starts + durations / 2, durations)
    pyplot.title('Vocalization Durations for {0}'.format('ALL' if speaker is None else speaker))
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Duration (s)')
    pyplot.subplot(2, 1, 2)
    pyplot.hist(durations, bins=numpy.logspace(0, 4, 100))
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.xlabel('Duration (s)')
    pyplot.ylabel('Count')


def plot_intervals(recording, speaker):
    i, starts, ends = recording.filter_speaker(speaker)
    intervals = starts[1:] - ends[:-1]
    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(starts[1:], intervals)
    pyplot.title('Vocalization Intervals for {0}'.format(speaker))
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Interval (s)')
    pyplot.subplot(2, 1, 2)
    pyplot.hist(intervals, bins=numpy.logspace(0, 4, 50))
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.xlabel('Interval (s)')
    pyplot.ylabel('Count')


def plot_volubility(recording, speaker):
    minutes = math.ceil((recording.ends[-1] - recording.starts[0]) / 60)
    volubility = numpy.zeros(minutes)
    i, starts, ends = recording.filter_speaker(speaker)
    for m in range(minutes):
        start_minute = 60 * m
        end_minute = 60 * m + 60
        for start, end in zip(starts, ends):
            volubility[m] += max(min(end_minute, end) - max(start_minute, start), 0)
    volubility /= 60
    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(60 * numpy.arange(minutes), volubility)
    pyplot.title('Volubility for {0}'.format(speaker))
    pyplot.xlabel('Time (min)')
    pyplot.ylabel('Vocalized Seconds / Minute')
    pyplot.subplot(2, 1, 2)
    pyplot.hist(volubility, bins=50)
    pyplot.yscale('log')
    pyplot.xlabel('Volubility')
    pyplot.ylabel('Count')
