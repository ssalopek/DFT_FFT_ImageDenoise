"""http://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm 

image_source = plt.imread('C:/FaksGit/FourierFilter/man.png').astype(float)

plt.figure()
plt.imshow(image_source, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title("Original image")

image_fft = fftpack.fft2(image_source)

def show_spectrum(image_fft):
    plt.imshow(np.abs(image_fft), norm=LogNorm(vmin=5))
    plt.colorbar() 

plt.figure()
show_spectrum(image_fft)
plt.title("Fourier transform")

keep_fraction = 0.1
image_fft2 = image_fft.copy()
row, col = image_fft2.shape

image_fft2[int(row*keep_fraction):int(row*(1-keep_fraction))] = 0
image_fft2[:, int(col*keep_fraction):int(col*(1-keep_fraction))] = 0

plt.figure()
show_spectrum(image_fft2)
plt.title('Filtered Spectrum')

image_new = fftpack.ifft2(image_fft2).real

"""
cv2.imshow('Reconstructed image FFT', image_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('moonlanding_FFT.png',image_new)
"""

fig = plt.figure()
plt.imshow(image_new, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title('Reconstructed Image FFT')
fig.savefig('manFFT.png')












