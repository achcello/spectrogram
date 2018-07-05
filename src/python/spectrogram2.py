import numpy as np
import matplotlib.pyplot as plt

# out of date
'''
class Signal(object):
	"""a signal class for organization"""
	def __init__(self, name):
		self.name = name
	
	name = 'signal'
	sampleRate = 0
	length = 0
	duration = 0
	values = np.array([])
'''

# other updated helper functions go here


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
	try:
		_ = signal.name
	except AttributeError as e:
		print('AttributeError: input is not a Signal object')

	print('Name:', signal.name)
	print('Length:', signal.length)
	print('Sample Rate:', signal.sampleRate)
	print('Duration:', signal.duration)
	print('Values:', signal.values)