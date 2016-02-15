import sys
import numpy
from scipy.io import wavfile
import sounddevice as sd


buckeye_root = '/media/tim/LittleDrive/buckeye/'


def get_phone_labels():
    starts = []
    types = []
    is_header = True
    for line in open(buckeye_root + 's01/s0101a/s0101a.phones', 'r'):
        if is_header:
            if line.startswith("#"):
                is_header = False
        else:
            fields = line.split()
            starts.append(float(fields[0]))
            types.append(fields[2].strip())
    starts = numpy.array(starts)
    durations = starts[2:] - starts[1:-1]
    return numpy.vstack((starts[1:-1], durations)), types


def get_recording():
    return wavfile.read(buckeye_root + 's01/s0101a/s0101a-wyes.wav')


def split_phones(times, sample_rate, recording):
    phones = []
    for time in times:
        start = time[0]
        duration = time[1]
        phones.append(recording[int(start * sample_rate):int(start + duration * sample_rate)])
    return phones

if __name__ == "__main__":
    times, labels = get_phone_labels()
    sample_rate, recording = get_recording()
    phones = split_phones(times, sample_rate, recording)
    for time, phone, label in zip(times, phones, labels):
        if (time[0] < 60):
            print("{0:.1f}s {1}".format(time[0], label))
            sd.play(numpy.array(phone), sample_rate, blocking=True)
