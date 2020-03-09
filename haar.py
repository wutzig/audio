import pyaudio as pa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import wave
import time

def FFT(x_in):
    x = np.asarray(x_in, dtype = np.float)
    N = x_in.shape[0]
    if(N % 2) > 0:
        raise ValueError("length of data must be power of 2")
    elif(N <= 32): #compute the Fourier transform for input length 32
        n = np.arange(N)
        k = n.reshape((N,1))
        A = np.exp(-2j * np.pi * k * n / N)
        return np.dot(A, x)
    else:
        theta_even  = FFT(x[0::2]) #FT of even input components
        theta_odd   = FFT(x[1::2]) #FT of odd input components
        factor      = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([ theta_even + factor[:int(N / 2)] * theta_odd,
                                theta_even + factor[int(N / 2):] * theta_odd] )

wf = wave.open('part.wav', 'rb')
p = pa.PyAudio()

POWER       = 11
SAMPLES     = 2 ** POWER
FORMAT      = p.get_format_from_width(wf.getsampwidth())
CHANNELS    = wf.getnchannels()
RATE        = wf.getframerate()
FRAMES      = wf.getnframes()

print( FORMAT )
print( RATE )
print( CHANNELS )

## Read in all the data and convert it to 16 bit integers
scales              = int(np.ceil( np.log2( FRAMES ) ))
signalLength        = 2 ** scales
x                   = np.arange( signalLength )
data            = np.linspace( 0, 0, signalLength )
data[:FRAMES]   = np.fromstring(wf.readframes(FRAMES), dtype = np.int16)[::CHANNELS]

print(scales)
wavelet_coeff = [[] for j in range(scales + 1)]
scaling_coeff = [[] for j in range(scales + 1)]
for resolution in range(1, scales + 1):
    size    = 2 ** resolution
    num     = 2 ** (scales - resolution)
    for t in range(num):
        coeff                               =  np.sum(data[t * size : (t+1) * size]) / size
        size2                               =  int(size / 2)
        left                                =  np.sum(data[t * size : (t+1) * size - size2])
        right                               =  np.sum(data[(t+1) * size - size2 : (t+1) * size])
        scaling_coeff[resolution].append( coeff )
        wavelet_coeff[resolution].append( (left - right) / size )

fig, ax    = plt.subplots(1)
ax.plot( range(signalLength), data )
plt.show()

new = scaling_coeff[scales][0] * np.zeros( signalLength )
print(len(scaling_coeff[scales]))
for resolution in range(scales, 0, -1):
    size    = 2 ** resolution
    num     = 2 ** (scales - resolution)
    for t in range(num):
        coeff = wavelet_coeff[resolution][t]
        size2 = int(size / 2)
        new[ t * size : (t + 1) * size  - size2]        += coeff
        new[ (t + 1) * size  - size2 : (t + 1) * size]  -= coeff
    fig, ax = plt.subplots(1)
    ax.plot(range(signalLength), new)
    plt.show()


## Compute the FFT of the data
# now                 = time.time()
# fft                 = FFT( data_int )
# print("FFT computed in ", time.time() - now, " seconds.")
# fft2                = np.sqrt(np.real(np.multiply(fft, np.conjugate(fft))))

# ## Plot the data and its Fourier transform
# fig1, (ax1, ax2)    = plt.subplots(2)
# line1               = ax1.plot( x, data_int )
# line2               = ax2.plot ( x[:int(signalLength / 2)], fft2[:int(signalLength / 2)] )
# plt.show( block = True )