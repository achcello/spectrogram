import numpy as np
import matplotlib.pyplot as plt

from signals import Signal, G_xx  # custom module


def spectrogram(signal, binWidth, overlap=1):
    """
    Generates the spectrogram of an input signal.

    @param signal The input signal object
    @return specs The values of the spectrogram
    @return f The frequency spectrum
    @return t The time domain
    """
    try:
        signal.name = signal.name
    except AttributeError as e:
        print('AttributeError: input is not a Signal object')

    f = np.linspace(0, binWidth // 2 * signal.sampleRate // binWidth, binWidth // 2)
    t = np.linspace(0, signal.length / signal.sampleRate, signal.length // binWidth * overlap)

    starts = np.arange(0, signal.length, binWidth // overlap)
    starts = np.append(starts, signal.length)
    specs = np.zeros((binWidth // 2, np.shape(t)[0]))

    for step in range(np.shape(starts)[0] - overlap - 1):
        subsignal = Signal(sampleRate=signal.sampleRate,
                           length=starts[step + overlap] - starts[step],
                           values=signal.values[starts[step]:starts[step + overlap]])
        specs[:, step] = G_xx(subsignal)

    return specs, f, t


if __name__ == '__main__':
    bird = Signal('Outside', '171013104635_1_B.wav')
    bird.generateValsFromFile()

    # plt.plot(bird.values)
    # plt.show()

    # plt.xlabel('Frequency (Hz)')
    # x = np.linspace(0, bird.length // 2 * bird.freqRes, bird.length // 2)
    # plt.ylabel('Intensity (some sort of 10*log10 thing)')
    # y = G_xx(bird)
    # plt.xlim(0, 5000)
    # plt.plot(x, y)
    # plt.show()

    plt.figure(figsize=(7, 5))
    plt.subplot(321)
    plt.title(bird.name)
    specs, f, t = spectrogram(bird, 1000, 200)
    specs = specs[:, ::20]
    t = t[::20]
    print('Heatmap size:', np.shape(specs))
    t, f = np.meshgrid(t, f)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 6000)
    plt.pcolormesh(t, f, specs, vmin=-20)
    # plt.colorbar()
    plt.savefig('../../figures/spec.png', dpi=300, bbox_inches='tight')
