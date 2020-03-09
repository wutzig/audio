import matplotlib.pyplot as plt
import wave
import pyaudio
import numpy as np

def FFT(x_in):
    x = np.asarray(x_in, dtype=np.float)
    N = len(x)
    print("Input vector has length", N)
    # TO DO: 
    # implentiere die schnelle Fourier Transformation rekursiv
    # überprüfe, ob die Länge N des Inputvektors eine Zweierpotenz ist, 
    if ( N % 2) > 0:
        raise ValueError("falsche Inputlänge")
    elif( N <= 32):
        n = np.arange(N)
        k = n.reshape((N,1))
        A = np.exp( -2j * np.pi * k * n / N)
        return np.dot(A, x)
    else:
        theta_even = FFT(x[0::2])
        theta_odd  = FFT(x[1::2])
        factor     = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate( [theta_even + factor[:int(N/2)] * theta_odd,
                                theta_even + factor[int(N/2):] * theta_odd ] )
    # sonst 'raise ValueError("falsche Inputlänge")'
    # falls N <= 32 (zum Bsp), bestimme die Fourier Matrix und berechne die diskrete FT nach Definition
    # falls N >  32 bestimme die geraden Komponenten des Inputvektors 'x_even = x[::2]'
    # und dessen FFT 'E = FFT(x_even)'
    # bestimme die ungeraden Komponenten 'x_odd = x[1::2]'
    # und dessen FFT 'O = FFT(x_odd)'
    # kombiniere beide Resultate in einen Vektor und gib diesen zurück

signal      = wave.open("440.wav")
FRAMES      = signal.getnframes()
CHANNELS    = signal.getnchannels()
RATE   = signal.getframerate()

print("frames:", FRAMES)
print("channels:", CHANNELS)
print("framerate:", RATE)
print("signal length:", FRAMES / RATE, "sec")

figure1, ax1    = plt.subplots(1)
data            = signal.readframes(FRAMES)

data_int        = np.fromstring(data, dtype = np.int16)[::CHANNELS]
signalLength    = 2 ** np.ceil(np.log2( FRAMES ))
signal_data     = np.linspace(0, 0, signalLength)
signal_data[:FRAMES] = data_int
signal_fft      = FFT( signal_data )
fft2            = np.sqrt(np.real(np.multiply(signal_fft, np.conjugate(signal_fft))))
line1           = ax1.plot(np.arange(signalLength), fft2)
plt.show()

# player = pyaudio.PyAudio()
# FORMAT = player.get_format_from_width(signal.getsampwidth())
# stream = player.open(
#     format   = FORMAT,
#     channels = CHANNELS,
#     rate     = int(RATE),
#     output = True )

# stream.write(data)
# FFT(data_int)