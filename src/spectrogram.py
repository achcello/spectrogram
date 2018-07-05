# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:33:53 2018

@author: Aryn Harmon
"""

import audioread
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import seaborn as sns


def hann(N):
    pass


def visualize(file):
    audio = audioread.audio_open(file)        # read in the audio file
    print(' = = = Audio Information = = = ')  # print out some information
    print('Channels:', audio.channels)
    channels = audio.channels
    print('Sample Rate:', audio.samplerate)
    sampleRate = audio.samplerate
    print('Duration:', audio.duration)

    audio = scipy.io.wavfile.read(file)[1]    # read from the converted file
    if channels > 1:
        audio = np.mean(audio, axis=1)        # collapse into one channel (?)
    audio = audio[100:4196]  # piano goes from 5000-100000
    samples = np.shape(audio)[0]
    duration = samples / sampleRate
    print("Trimmed Samples:", samples)
    print("Trimmed Duration:", duration)

    audioFFT = 10*np.log10(np.fft.fftshift(np.fft.fft(audio, norm="ortho"))/ sampleRate)  # FFT of the clip, piano is -20000

    sns.set()
    plt.figure(1)
    plt.plot(audio)
    plt.figure(2)
    # plt.ylim(-1000000, 1000000)
    # plt.plot(np.matrix.transpose(audioFFT)[1:], 'r-')  # piano is -5000

    bucketWidth = 512
    buckets = samples // bucketWidth          # calculate number of buckets
    spec = []                                 # accumulator arroy
    for bucket in range(buckets * 1):
        start = (bucket * bucketWidth) // 1
        end = start + bucketWidth
        # print(start, end)
        spec.append(10*np.log10(.02*np.fft.fft(audio[start:end],
                                               norm="ortho")[1:10]/(sampleRate*bucketWidth)))
    plt.figure(3)
    sns.heatmap(np.matrix.transpose(np.abs(np.array(spec)))).invert_yaxis()


if __name__ == '__main__':
    file = '440SineWave.wav'
    visualize(file)
