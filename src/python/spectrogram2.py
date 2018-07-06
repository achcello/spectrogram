import numpy as np
import matplotlib.pyplot as plt
import audioread


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

    def __init__(self, name, filePath='', sampleRate=0, length=0, values=np.array([])):
        """
        Signal contructor

        @param name The name given to the signal
        @param filePath The path where the signal can be found
        @param sampleRate The sampling rate of the signal
        @param length The number of samples in the signal
        @param values A numpy array of values representing the signal
        """
        self.name = name
        self.filePath = filePath
        self.sampleRate = sampleRate
        self.length = length
        self.values = values

    def generateValsFromFile(self):
        if self.filePath == '':
            print('Please first set the filepath for the signal source.')
            break
        audio = audioread.audio_open(self.filePath)  # read in the audio file
        self.sampleRate = audio.samplerate
        channels = audio.channels
        audio = scipy.io.wavfile.read(self.filePath)[1]  # read from the converted file
        if channels > 1:
            audio = np.mean(audio, axis=1)  # collapse into one channel (?)
        self.length = np.shape(audio)[0]
        self.values = audio

    def getDuration(self):
        """
        Getter for the duration of the signal

        @return The duration of the signal in seconds (presumably)
        """
        return self.length / self.sampleRate


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


def linearSpec(signal):
    """
    Get the linear spectrum, $X_n(f)$, for a given signal, $f$, of a signal.
    """
    X_n = np.fft.fft(signal.values) * 

def spectrogram(signal):
    """
    Function to generate the spectrogram of an input signal using numpy tools

    @param signal The input signal object
    @return f The frequency spectrum
    @return t The time domain
    @return Sxx The values of the spectrogram
    """
    try:
        _ = signal.name
    except AttributeError as e:
        print('AttributeError: input is not a Signal object')

    print('Name:', signal.name)
    print('Length:', signal.length)
    print('Sample Rate:', signal.sampleRate)
    print('Duration:', signal.duration)
    print('Values:', signal.values)

    Sxx = 