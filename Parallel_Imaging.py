# import all nessersary modules/libraries
import imageio.v2 as iio
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom  # import test image
import numpy as np
from scipy import fftpack
from scipy import signal
import GRAPPA
import Recon_functions

# import test image, convert to numpy array and find dimensions
img = shepp_logan_phantom()
img_np = np.array(img) * 100
Ny,Nx = np.shape(np.array(img))

plt.figure()
plt.imshow(img_np)

## Coil Implementation

# coil parameters
n_coils = 8
nx_coils = 2
# coil_x_pos = [100,200]
coil_x_pos = [150,250]
# print(coil_x_pos)
ny_coils = 4
# coil_y_pos = np.linspace(50,200,ny_coils)
coil_y_pos = np.linspace(50,350,ny_coils)
sigma = 50

# Create spatial grid
x = np.linspace(0, Nx, Nx)
y = np.linspace(0, Ny, Ny)
X, Y = np.meshgrid(x, y)

# assigning the coordiantes of each coils position
coil_pos = []
for i in coil_y_pos:
    for j in coil_x_pos:
        coil_pos.append((i,j))
# print("coil_pos ",coil_pos)
coil_pos = np.array(coil_pos)

# assigning coil sensitivity with Gaussian distribution originating from coil position
coil_sensitivities = []
for y0,x0 in coil_pos:
    # calculating the Gaussian distibution at certain coil locations with (imaginary) phase distribution
    G_r = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * (sigma**2)))
    G1im = 1j * np.exp(-((X-x0-5)**2 + (Y-y0+10)**2) / (2 * (sigma**2)))
    coil_sensitivities.append(G_r + G1im)
coil_sensitivities = np.array(coil_sensitivities,dtype=complex)

# plotting sensitivities
plt.figure()
plt.suptitle('Coil sensitivities', fontsize=16)
for i,j in enumerate(coil_sensitivities):
    plt.subplot(4,4,i+1)
    plt.imshow(abs(j))

# multiply the image by the sensitivities
img_np = np.array(img)
coil_view = []
plt.figure()
plt.suptitle('Coil view image domain', fontsize=16)
for i in enumerate(coil_sensitivities):
    # each individual coil view
    x = np.multiply(i[1],img_np)
    coil_view.append(x)
    plt.subplot(4,4,i[0]+1)
    plt.imshow(abs(x))
coil_view = np.array(coil_view,dtype=complex)

## K-Sapce transformations

# fourier transform image and sensitivities to get K-Space raw data
plt.figure()
plt.suptitle('Coil view K-Space', fontsize=16)
img_kspace = np.fft.fftshift(np.fft.fft2(img_np))
coil_view_kspace = []
for index,i in enumerate(coil_view):
    x = np.fft.fftshift(np.fft.fft2(i))
    coil_view_kspace.append(x)
    plt.subplot(4,4,index+1)
    plt.imshow(abs(np.log(x)))
    plt.colorbar()
coil_view_kspace = np.array(coil_view_kspace,dtype=complex)

#plt.imshow(abs(np.log(img_kspace)))

print(len(coil_view_kspace[0]))

## Parallel imaging: Sampling K-Space coil views with ACS

# acceleration rate (number of phase encoding lines sampled)
R = 2

# ACS region (for each coil)
ACS_num = 24
abv = 12
bel = 11
coil_ACS = []
coil_under_sampled = []

# initialise the sampling of the image
sampled_coils_ks = []

# sampling K-Space for each coil (with ACS)
for i in coil_view_kspace:
    # full K-Space sampling
    x = np.zeros_like(i, dtype=complex)
    y = np.zeros_like(i, dtype=complex)
    x[::R] = i[::R]
    y[::R] = i[::R]
    coil_under_sampled.append(y)
    row = 199
    # including ACS lines with K-Space
    row_min = row - abv
    row_max = row + bel
    x[row_min:row_max + 1:1, :] = i[row_min:row_max + 1:1, :]
    coil_ACS.append(i[row_min:row_max + 1:1, :])
    sampled_coils_ks.append(x)

sampled_coils_ks = np.array(sampled_coils_ks)
coil_under_sampled = np.array(coil_under_sampled)
coil_ACS = np.array(coil_ACS)
# Inverse Fourier Transform and plotting for each aliased coil view
for index, i in enumerate(sampled_coils_ks):
    img_samp = np.fft.ifft2(np.fft.ifftshift(i))
    img_kspace = i
    plt.subplot(4, 4, index + 1)
    # plt.imshow(abs(img_samp))
    # plt.imshow(abs(img_kspace),vmin=0,vmax=1000)
    plt.imshow(abs(np.log(img_kspace + 1 * 10 ** -6)))
    plt.colorbar()

# defining the kernel size for the GRAPPA reconstruction
kernel_size = [2, 3]

## perform GRAPPA reconstruction
g2 = GRAPPA.GRAPPA(kernel_size, R, coil_ACS, sampled_coils_ks)

# show the final GRAPPA reconstructed image
plt.figure()
plt.suptitle('Sum-of-square full GRAPPA reconstruction', fontsize=16)
plt.imshow(abs(g2.image))

plt.show()
