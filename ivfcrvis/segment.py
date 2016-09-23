from matplotlib import pyplot
from python_speech_features import mfcc, logfbank
import numpy


class Segment:
    """Segment represents an interval within a longer recording in which a speaker
    can be identified. This object provides several methods of acoustic analysis."""

    def __init__(self, recording, speaker, i):
        """Construct a new segment from the given recording by cutting out the
        ith interval for the selected speaker."""
        self.start = recording.starts[i]
        self.end = recording.ends[i]
        self.speaker = speaker
        self.i = i
        self.sample_rate, self.signal = recording.read_segment(speaker, i)
        self.samples = len(self.signal)
        self.duration = len(self.signal) / self.sample_rate

    def power(self, window_size, step_size):
        """Return a time series corresponding to the acoustic power of the segment waveform within
        each window of the given size, offset by the specified step."""
        steps = numpy.arange(window_size, self.signal.size, step_size)
        power = numpy.zeros(len(steps))
        for i, step in zip(range(len(steps)), steps):
            windowed_signal = self.signal[step-window_size:step].astype(int)
            rms = numpy.sqrt(numpy.mean(numpy.square(windowed_signal)))
            power[i] = 10 * numpy.log10(rms)
        return steps / float(self.sample_rate), power

    def power_spectrum(self, window_size, step_size):
        """Return a time series of power spectra, where each spectrum corresponds to the acoustic power within a window
        of the given size and offset from the previous window by the given step. The spectra represent the power over
        frequency within the window, where the associated frequencies are specified by the first return value."""
        frequencies = numpy.fft.rfftfreq(window_size, d=1./self.sample_rate)
        spectrum = numpy.ndarray((0, frequencies.size))
        window = numpy.hanning(window_size)
        for i in range(window_size, self.signal.size, step_size):
            x = window * self.signal[i-window_size:i]
            spectrum = numpy.vstack((spectrum, abs(numpy.fft.rfft(x).real)))
        return frequencies, spectrum

    def formants(self, window_size, step_size, count):
        """Return a time series of formant frequencies identified from the power spectra. Count is the number of
        formants to identify."""
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

    def frequencybank(self, window_size=0.05, step_size=0.05, truncate=0.6):
        """Returns a Mel-frequency filter bank for this segment."""
        x = self.signal
        if truncate:
            x = x[:int(truncate * self.sample_rate)]
        return logfbank(x, self.sample_rate, winlen=window_size, winstep=step_size)

    def mfccs(self):
        """Returns the Mel-Frequency Cepstral Coefficients for this segment."""
        return mfcc(self.signal[:int(0.6 * self.sample_rate)], self.sample_rate, winlen=0.05, winstep=0.05, numcep=40, nfilt=80)


def plot_segment(segment, window_size, step_size):
    """Plots the waveform, power over time, spectrogram with formants, and power over frequency of a Segment."""
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


def plot_sample_mfccs(recording, speaker='FAN', n=4):
    """Plots the MFCCs of the first n segments for the selected speaker."""
    index = numpy.where(numpy.array(recording.speakers) == speaker)[0]
    pyplot.figure()
    for i,j in zip(index[:n], range(1, n + 1)):
        segment = Segment(recording, speaker, i)
        pyplot.subplot(2, n / 2, j)
        pyplot.title('Vocalization {0} MFCC'.format(i))
        pyplot.imshow(segment.mfccs().transpose(), extent=[0, 600, 0, 13], aspect='auto', interpolation='nearest',
                      cmap=pyplot.cm.get_cmap('Spectral'))
        pyplot.xlabel('Time (ms)')
        pyplot.ylabel('Coefficient')
        pyplot.colorbar()
