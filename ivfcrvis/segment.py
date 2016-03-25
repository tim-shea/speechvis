from ivfcrvis.recording import Recording
from matplotlib import pyplot
from features import mfcc
import numpy
from scipy.signal import savgol_filter


class Segment:
    def __init__(self, recording, speaker, i):
        self.start = recording.starts[i]
        self.end = recording.ends[i]
        self.speaker = speaker
        self.i = i
        self.sample_rate, self.signal = recording.read_segment(speaker, i)
        self.samples = len(self.signal)
        self.duration = len(self.signal) / self.sample_rate

    def power(self, window_size, step_size):
        steps = numpy.arange(window_size, self.signal.size, step_size)
        power = numpy.zeros(len(steps))
        for i, step in zip(range(len(steps)), steps):
            windowed_signal = self.signal[step-window_size:step].astype(int)
            rms = numpy.sqrt(numpy.mean(numpy.square(windowed_signal)))
            power[i] = 10 * numpy.log10(rms)
        return steps / float(self.sample_rate), power

    def power_spectrum(self, window_size, step_size):
        frequencies = numpy.fft.rfftfreq(window_size, d=1./self.sample_rate)
        spectrum = numpy.ndarray((0, frequencies.size))
        window = numpy.hanning(window_size)
        for i in range(window_size, self.signal.size, step_size):
            x = window * self.signal[i-window_size:i]
            spectrum = numpy.vstack((spectrum, abs(numpy.fft.rfft(x).real)))
        return frequencies, spectrum

    def formants(self, window_size, step_size, count):
        frequencies, spectrum = self.power_spectrum(window_size, step_size)
        formants = numpy.full((spectrum.shape[0], count), float('NaN'))
        for step in range(spectrum.shape[0]):
            #mean_spectrum = numpy.mean(spectrum, axis=0)
            mean_spectrum = spectrum[step]
            peaks = []
            for x, y in zip(range(len(mean_spectrum)), mean_spectrum):
                if y == max(mean_spectrum[max(0, x - 10):min(len(mean_spectrum), x + 10)]):
                    peaks.append((x, y))
            peaks = numpy.array(peaks)
            peaks = peaks[numpy.argsort(peaks[:,1]),:]
            f = numpy.sort(frequencies[peaks[-count:,0].astype(int)])
            f = numpy.pad(f, (0, count - f.size), 'constant', constant_values=float('NaN'))
            formants[step,:] = f
        return formants

    def mfccs(self):
        return mfcc(self.signal[:int(0.6 * self.sample_rate)], self.sample_rate, winlen=0.05, winstep=0.05, numcep=40, nfilt=80)
        #return mfcc(self.signal[:int(0.6 * self.sample_rate)], self.sample_rate)


def plot_segment(segment, window_size, step_size):
    pyplot.figure()
    pyplot.subplot(2, 2, 1)
    pyplot.plot(numpy.linspace(0, segment.duration, segment.samples), segment.signal)
    pyplot.xlim(0, segment.duration)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Sound Pressure')
    pyplot.subplot(2, 2, 2)
    steps, power = segment.power(window_size, step_size)
    pyplot.plot(steps, power)
    pyplot.xlim(0, segment.duration)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Power (dB)')
    pyplot.subplot(2, 2, 3)
    pyplot.specgram(segment.signal, NFFT=window_size, Fs=segment.sample_rate, noverlap=step_size)
    formants = segment.formants(window_size, step_size, 2)
    pyplot.plot(numpy.linspace(0, segment.duration, len(formants)), formants, 'o')
    pyplot.xlim(0, segment.duration)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Frequency (Hz)')
    pyplot.subplot(2, 2, 4)
    frequencies, spectrum = segment.power_spectrum(window_size, step_size)
    pyplot.plot(frequencies / 1000, 10 * numpy.log10(numpy.mean(spectrum, axis=0)))
    pyplot.xlabel('Frequency (kHz)')
    pyplot.ylabel('Power (dB)')


def plot_sample_mfccs(recording):
    index = numpy.where(numpy.array(recording.speakers) == 'FAN')[0]
    pyplot.figure()
    for i,j in zip(index[:4], range(1, 5)):
        segment = Segment(recording, 'FAN', i)
        pyplot.subplot(2, 2, j)
        pyplot.title('Vocalization {0} MFCC'.format(i))
        pyplot.imshow(segment.mfccs().transpose(), extent=[0, 600, 0, 13], aspect='auto', interpolation='nearest', cmap=pyplot.cm.get_cmap('Spectral'))
        pyplot.xlabel('Time (ms)')
        pyplot.ylabel('Coefficient')
        pyplot.colorbar()
