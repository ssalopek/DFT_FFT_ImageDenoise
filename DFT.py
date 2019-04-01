import numpy as np
import cv2
from matplotlib import pyplot as plt

img = plt.imread('C:/FaksGit/FourierFilter/lenna.png').astype(float)
img_float32 = np.float32(img)

plt.figure()
plt.title('Original image')
plt.imshow(img, plt.cm.gray)
plt.xticks([]), plt.yticks([])

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = rows/2 , cols/2  # center

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[int(crow)-30:int(crow)+30, int(ccol)-30:int(ccol)+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

"""
cv2.imshow('Reconstructed image DFT', img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('moonlanding_DFT.png',img_back)
"""
"""
plt.imshow(img_back, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
plt.savefig('moonlandingDFT.png')
"""

fig = plt.figure()
plt.imshow(img_back, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title('Reconstructed Image DFT')
plt.show()
fig.savefig('lennaDFT.png')























