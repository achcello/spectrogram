"""
Finds the cross correlation between two signals.
"""
import numpy as np
import matplotlib.pyplot as plt

from signals import Signal, linearSpectrum
from spectrogram import spectrogram


def S_xy(signal1, signal2):
    """
    This method generates the cross-spectral density of two given signals.

    @return sxy: the cross-spectral density
    """
    linSpecX = linearSpectrum(signal1)
    linSpecY = linearSpectrum(signal2)
    sxy = 1 / signal1.getDuration() * np.conj(linSpecX) * linSpecY
    return sxy


def crossCorrelation(signal1, signal2, binWidth, overlap):
    """
    Generate the cross-correlation between two signals.

    @returns: the cross-correlation
    """
    if signal1.length != signal2.length:
        raise ValueError("signals are not the same length")
    if signal1.sampleRate != signal2.sampleRate:
        raise ValueError("signals do not have the same sample rate")

    t = np.linspace(0, signal1.length / signal1.sampleRate, signal1.length * overlap)
    print(t)

    starts = np.arange(0, signal1.length, binWidth // overlap)
    starts = np.append(starts, signal1.length)
    corr = np.zeros(signal1.length * overlap)
    '''
    for step in range(1, np.shape(starts)[0]):
        subsignal1 = Signal(sampleRate=signal1.sampleRate,
                            length=starts[step + overlap] - starts[step],
                            values=signal1.values[starts[step - 1]:starts[step]])
        subsignal2 = Signal(sampleRate=signal2.sampleRate,
                            length=starts[step + overlap] - starts[step],
                            values=signal2.values[starts[step - 1]:starts[step]])
        corr[starts[step]:starts[step+overlap]] = np.fft.ifft(S_xy(subsignal1, subsignal2)) * signal1.sampleRate
    '''
    corr = np.fft.ifft(S_xy(signal1, signal2)) * signal1.sampleRate
    corr = corr / np.max(corr)
    return corr, t
    # return corr, t


if __name__ == "__main__":
    # create signals from files
    chirp = Signal('chirp', 'LFM_1K_5K.wav')
    chirp.generateValsFromFile()
    chirp.values /= np.max(chirp.values)
    chirp.values -= np.mean(chirp.values)
    outside = Signal('outside', '171013104635_1_B.wav')
    outside.generateValsFromFile()
    outside.values /= np.max(outside.values)

    plt.figure(figsize=(17*2/3, 22*2/3))
    plt.tight_layout()

    # plot signals' spectrograms

    plt.subplot(321)
    plt.title("Chirp")
    specs, f, t = spectrogram(chirp, 1000, 200)
    specs = specs[:, ::20]
    t = t[::20]
    print('Heatmap size:', np.shape(specs))
    t, f = np.meshgrid(t, f)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 6000)
    plt.xlim(0, 0.25)
    plt.pcolormesh(t, f, specs, vmin=-150, cmap="BuPu")
    print("@debug created first spectrogram")

    plt.subplot(322)
    plt.title("Outside")
    specs2, f2, t2 = spectrogram(outside, 1000, 200)
    specs2 = specs2[:, ::20]
    t2 = t2[::20]
    print('Heatmap size:', np.shape(specs2))
    t2, f2 = np.meshgrid(t2, f2)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 6000)
    plt.pcolormesh(t2, f2, specs2, vmin=-100, cmap="BuPu")
    print("@debug created second spectrogram")

    # plot signals' time series
    tm = np.linspace(0, outside.getDuration(), outside.length)
    plt.subplot(323)
    plt.title("Chirp")
    plt.xlabel("Time (s)")
    plt.plot(tm[::10], chirp.values[::10])
    print("@debug created first time series")
    plt.subplot(324)
    plt.title("Outside Recording")
    plt.xlabel("Time (s)")
    plt.plot(tm[::10], outside.values[::10])
    print("@debug created second time series")

    # plot signals' cross correlation
    plt.subplot(313)
    plt.xlabel("Time (s)")
    plt.title("Cross-correlation")
    rxy, tm = crossCorrelation(chirp, outside, 2000, 1)
    plt.plot(tm[::10], rxy[::10])
    print("@debug created cross-correlation")

    plt.savefig('../../figures/corr.png', dpi=300, bbox_inches='tight')
