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

wf = wave.open('440.wav', 'rb')
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

### Read in all the data and convert it to 16 bit integers
# signalLength        = 2 ** np.ceil( np.log2( FRAMES ) )
# x                   = np.arange( signalLength )
# data_int            = np.linspace( 0, 0, signalLength )
# data_int[:FRAMES]   = np.fromstring(wf.readframes(FRAMES), dtype = np.int16)[::CHANNELS]

# ## Compute the FFT of the data
# now                 = time.time()
# fft                 = FFT( data_int )
# print("FFT computed in ", time.time() - now, " seconds.")
# fft2                = np.sqrt(np.real(np.multiply(fft, np.conjugate(fft))))

# ## Plot the data and its Fourier transform
# fig1, (ax1, ax2)    = plt.subplots(2)
# line1               = ax1.plot( x, data_int )
# line2               = ax2.plot ( x, fft2 )
# plt.show( block = True )

### Rewind the signal to read again from beginning
wf.rewind()


### Now consider samples of the signal with length SAMPLES = 2 ** POWER
### and compute the DFT for every sample
fig1, (ax1, ax2) = plt.subplots(2)

line1 = ax1.plot( np.arange(SAMPLES), np.zeros(SAMPLES) )[0]
ax1.set_xlim( 0, SAMPLES )
ax1.set_ylim( -5000, 5000 ) # -5000, 5000

line2 = ax2.semilogx( np.arange(2 ** (POWER - 1) ), np.zeros(2 ** (POWER - 1)), '-r' )[0]
ax2.set_xlim( 1, 2 ** (POWER - 1))
ax2.set_ylim( 0, 120_000 ) #0, 100,000

plt.show(block = False)

stream = p.open(format      = FORMAT,
                channels    = CHANNELS,
                rate        = int(RATE),
                output      = True)

x       = np.arange( 0, SAMPLES )
theta   = np.logspace( 0, POWER - 1, num = SAMPLES / 2, base = 2.0 )

### the DFT matrix F
F       = np.exp( -2j * np.pi * np.outer(x, theta) / SAMPLES ) / np.sqrt(SAMPLES) #x, theta

def update(self):
    data = wf.readframes(SAMPLES)
    stream.write(data)
    data_int = np.fromstring(data, dtype = np.int16)[::CHANNELS]
    line1.set_ydata(data_int)
    fft     = np.dot( data_int, F )
    
    line2.set_ydata(np.sqrt(np.real(np.multiply(fft, np.conjugate(fft)))))
    return line1, line2,

ani = FuncAnimation( fig1,
                    func        = update,
                    frames      = int(FRAMES / SAMPLES) - 2,
                    interval    = 0,
                    blit        = True,
                    repeat      = False )
plt.show()

stream.stop_stream()
stream.close()

p.terminate()