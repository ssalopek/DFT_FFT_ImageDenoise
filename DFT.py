import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

start = time.time()
#Load source image
image_source = plt.imread('C:/FaksGit/FourierFilter/TestImages/moonlanding.png').astype(float)
img_float32 = np.float32(image_source)

#Plot input image
plt.figure()
plt.imshow(image_source, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title('Original image')

#Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT) 
#Shift the zero-frequency component to the center of the spectrum.
dft_shift = np.fft.fftshift(dft) 

rows, cols = image_source.shape
crow, ccol = rows/2 , cols/2  # centar

keep_fraction = 60

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[int(crow)-keep_fraction:int(crow)+keep_fraction, int(ccol)-keep_fraction:int(ccol)+keep_fraction] = 1

# apply mask and inverse DFT
fshift = dft_shift * mask

#The inverse of fftshift. Although identical for even-length x, 
#the functions differ by one sample for odd-length x
f_ishift = np.fft.ifftshift(fshift)

#Calculates the inverse Discrete Fourier Transform of a 1D or 2D array.
img_back = cv2.idft(f_ishift)

#Calculates the magnitude of 2D vectors
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

end = time.time()
print("Time:", end-start)

fig = plt.figure(figsize=(20,10))
plt.imshow(img_back, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title('Reconstructed Image DFT')
plt.show()
fig.savefig('baboon_gauss60_DFT60.png')























