import numpy as np
import matplotlib.pyplot as plt
import audioread
import scipy.io.wavfile


class Signal(object):
    """
    Class to represent signals as a python object
    """
    name = 'signal'
    filePath = '.'
    sampleRate = 0
    length = 0
    values = np.array([])
    start = 0  # start marker for analysis
    stop = 0  # stop marker for analysis
    freqRes = 0  # frequency resolution resulting from transformation

    def __init__(self, name, fileName='', sampleRate=0, length=0,
                 values=np.array([])):
        """
        Signal contructor

        @param name The name given to the signal
        @param filePath The name at which the signal can be found (the signal
        must be in the resources directory)
        @param sampleRate The sampling rate of the signal
        @param length The number of samples in the signal
        @param values A numpy array of values representing the signal
        """
        self.name = name
        self.filePath = '../../resources/' + fileName
        self.sampleRate = sampleRate
        self.length = length
        self.values = values

    def generateValsFromFile(self):
        if self.filePath == '':
            print('Please first set the filepath for the signal source.')
            return
        audio = audioread.audio_open(self.filePath)  # read in the audio file

        self.sampleRate = audio.samplerate
        channels = audio.channels
        audio = scipy.io.wavfile.read(self.filePath)[1]  # read from the file
        if channels > 1:
            audio = np.mean(audio, axis=1)  # collapse into one channel (?)
        self.values = np.double(audio) - 128
        self.values = self.values[:]
        self.length = np.shape(self.values)[0]
        self.values = hann(self.length) * self.values
        self.freqRes = self.sampleRate / self.length
        
        scaled = np.int16(self.values/np.max(np.abs(self.values)) * 32767)
        scipy.io.wavfile.write('window.wav', self.sampleRate, scaled)

        print('Name:', self.name)
        print('Length:', self.length)
        print('Sample Rate:', self.sampleRate)
        print('Duration:', self.getDuration())
        print('Values:', self.values)

    def getDuration(self):
        """
        Getter for the duration of the signal

        @return The duration of the signal in seconds (presumably)
        """
        if self.sampleRate == 0:
            print("Sample rate is zero. Pls don't make me divide by zero.")
            print('Duration is None.')
            return None
        if self.length == 0:
            print('Length = 0. You probably forgot to set it, or did not'
                  + ' generate it from the WAV file.')
        return self.length / self.sampleRate

    def linearSpectrum(self):
        """
        Get the linear spectrum, $X_n(f)$, for a given signal, $f$, of a
        signal.

        @return X_m: linear spectrum
        """
        X_m = np.fft.fft(self.values) / self.sampleRate
        return X_m

    def S_xx(self):
        """
        Given a linear spectrum and a window, generate the double-sided
        spectral density.

        @return dssd: double-sided spectral density
        """
        linSpec = self.linearSpectrum()
        dssd = 1 / self.getDuration() * np.conj(linSpec) * linSpec
        return dssd

    def G_xx(self):
        """
        Using the double-sided spectral density, generate the single-sided
        spectral density.

        @return sssd: single-sided spectral density
        """
        doubleSpec = self.S_xx()
        start = np.shape(doubleSpec)[0] // 2
        sssd = 2 * doubleSpec[start:]
        sssd[0], sssd[-1] = 0, 0
        return sssd


def hann(N):
    """
    Function for creating a Hann window for smoothing signal down to zero at
    the edges of the window.
    """
    if N < 1:
        return np.array([])
    elif N == 1:
        return np.ones(1)
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))


def spectrogram(signal):
    """
    Generates the spectrogram of an input signal.

    @param signal The input signal object
    @return f The frequency spectrum
    @return t The time domain
    @return Sxx The values of the spectrogram
    """
    try:
        signal.name = signal.name
    except AttributeError as e:
        print('AttributeError: input is not a Signal object')


if __name__ == '__main__':
    bird = Signal('bird', 'bird.wav')
    bird.generateValsFromFile()
    
    plt.plot(bird.values)
    plt.show()
    
    plt.xlabel('Frequency (Hz)')
    x = np.linspace(0, bird.length // 2 * bird.freqRes, bird.length // 2)
    plt.ylabel('Intensity (?)')
    y = bird.G_xx()

    # plt.plot(bird.linearSpectrum()[1:])
    plt.plot(x, y)
