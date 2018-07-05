# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:59:31 2018

@author: ach370
"""

import numpy as np
import matplotlib.pyplot as plt


def f(n):
    A = 1
    m_0 = 10
    N = 1600
    return A * np.cos(2*np.pi * m_0 * n / N)


n = np.arange(1600)
print(n)
print(f(n))
plt.subplot(211)
plt.plot(n, f(n))
fft = np.fft.fft(f(n))
plt.subplot(212)
plt.plot(fft)
