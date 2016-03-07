from ivfcrvis.lena import LenaRecording
from matplotlib import pyplot
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
            formants[step,:peaks.size] = numpy.sort(frequencies[peaks[-count:,0].astype(int)])
        return formants


def plot_segment(segment):
    pyplot.figure()
    pyplot.subplot(3, 1, 1)
    pyplot.plot(numpy.linspace(0, segment.duration, segment.samples), segment.signal)
    pyplot.xlim(0, segment.duration)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Sound Pressure')
    pyplot.subplot(3, 1, 2)
    pyplot.specgram(segment.signal, NFFT=128, Fs=segment.sample_rate, noverlap=64)
    pyplot.xlim(0, segment.duration)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Frequency (Hz)')
    pyplot.subplot(3, 1, 3)
    frequencies, spectrum = segment.power_spectrum(128, 128)
    pyplot.plot(frequencies / 1000, 10 * numpy.log10(numpy.mean(spectrum, axis=0)))
    pyplot.xlabel('Frequency (kHz)')
    pyplot.ylabel('Power (dB)')
