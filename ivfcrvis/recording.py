import numpy
import os
import math
from xml.etree import ElementTree
from scipy.io import wavfile
from matplotlib import pyplot
from features import logfbank


class Recording:
    """Recording reads an ITS file exported from LENA and parses out data about the segments and speakers in the
    corresponding WAV file. It also contains a method to split and save out individual segments as WAV files for
    acoustic analysis."""

    def __init__(self, root, recording_id):
        """Construct a new Recording by reading the ITS file in the directory root with a filename derived from
        recording_id."""
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
        self.samplerate = None
        self.signal = None
        self.duration = None

    def read_recording(self):
        """Read the WAV file corresponding to this Recording. This is deferred because it can be slow."""
        filepath = os.path.join(self.root, '{0}.wav'.format(self.recording_id))
        self.samplerate, self.signal = wavfile.read(filepath)
        self.duration = len(self.signal) / self.samplerate
    
    def frequency_banks(self, blockSize=600):
        if self.signal is None:
            self.read_recording()
        fbanks = numpy.zeros((0, 1, 26))
        start = 0
        while start < len(self.signal):
            end = start + blockSize * self.samplerate
            end = end if end < len(self.signal) else len(self.signal)
            block = self.signal[start:end]
            fbank = logfbank(block, self.samplerate, winlen=0.05, winstep=0.025)
            fbanks = numpy.concatenate((fbanks, numpy.reshape(fbank, (len(fbank), 1, 26))))
            start = end
        return fbanks

    def split_segments(self):
        """Split the WAV file for this recording into individual segments and save those segments in a directory
        structure according to the identified speaker."""
        recording_dir = os.path.join(self.root, self.recording_id)
        if not os.path.exists(recording_dir):
            os.makedirs(recording_dir)
        for speaker in set(self.speakers):
            speaker_dir = os.path.join(recording_dir, speaker)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)
        if self.signal is None:
            self.read_recording()
        for start, end, speaker, i in zip(self.starts, self.ends, self.speakers, range(len(self.starts))):
            segment = self.signal[int(start * self.samplerate):int(end * self.samplerate)]
            wavfile.write(os.path.join(recording_dir, speaker, '{0}.wav'.format(i)), self.samplerate, segment)

    def read_segment(self, category, i):
        """Read an individual segment WAV file. Returns the sample rate and signal."""
        filename = os.path.join(self.root, self.recording_id, category, '{0}.wav'.format(i))
        return wavfile.read(filename)

    def filter_speaker(self, speaker):
        """Return the indices, start times, and end times of all segments labeled with the speaker."""
        index = numpy.array(self.speakers) == speaker
        return numpy.where(index)[0], self.starts[index], self.ends[index]


def parse_time(formatted):
    """Returns the time in seconds indicated by the formatted string."""
    # TODO: This should not require Pacific timezone, lookup lena format spec
    if formatted.startswith('PT') and formatted.endswith('S'):
        return float(formatted[2:-1])


def plot_speaker_counts(recording):
    """Plot the number of segments in the recording for each speaker."""
    speakers, counts = numpy.unique(recording.speakers, return_counts=True)
    fig = pyplot.figure()
    pyplot.bar(numpy.arange(len(speakers)) + 0.1, counts)
    pyplot.title('Number of Vocalizations by Speaker')
    pyplot.xticks(numpy.arange(len(speakers)) + 0.5, speakers)
    pyplot.xlim(0, len(speakers))
    pyplot.xlabel('Speaker')
    pyplot.ylabel('Count')
    return fig


def plot_durations(recording, speaker=None):
    """Plot a time series and a histogram of segment durations, optionally filtered for a speaker."""
    if speaker is None:
        starts = recording.starts
        ends = recording.ends
    else:
        i, starts, ends = recording.filter_speaker(speaker)
    durations = ends - starts
    fig = pyplot.figure()
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
    return fig


def plot_intervals(recording, speaker):
    """Plot a time series and histogram of segment intervals labeled as speaker."""
    i, starts, ends = recording.filter_speaker(speaker)
    intervals = starts[1:] - ends[:-1]
    fig = pyplot.figure()
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
    return fig


def plot_volubility(recording, speaker):
    """Plot the volubility ratio (proportion of time that speaker is speaking) as a time series and histogram. This
    analysis uses one minute blocks to aggregate segments."""
    minutes = math.ceil((recording.ends[-1] - recording.starts[0]) / 60)
    volubility = numpy.zeros(minutes)
    i, starts, ends = recording.filter_speaker(speaker)
    for m in range(minutes):
        start_minute = 60 * m
        end_minute = 60 * m + 60
        for start, end in zip(starts, ends):
            volubility[m] += max(min(end_minute, end) - max(start_minute, start), 0)
    volubility /= 60
    fig = pyplot.figure()
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
    return fig
