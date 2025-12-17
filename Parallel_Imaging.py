# import all nessersary modules/libraries
import imageio.v2 as iio
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom  # import test image
import numpy as np
from scipy import fftpack
from scipy import signal
import GRAPPA
import Recon_functions

# import test image, convert to numpy array and find dimensions
#img = iio.imread("childMRI.png")
img = shepp_logan_phantom()
img_np = np.array(img)
Ny,Nx = np.shape(np.array(img))

plt.figure()
plt.imshow(img_np)

## Coil Implementation

# coil parameters
n_coils = 8
nx_coils = 2
#coil_x_pos = [100,200]
coil_x_pos = [150,250]
# print(coil_x_pos)
ny_coils = 4
#coil_y_pos = np.linspace(50,200,ny_coils)
coil_y_pos = np.linspace(50,350,ny_coils)
sigma = 50

# creating coils
coil_sensitivities = Recon_functions.creating_coils(Nx, Ny, coil_y_pos, coil_x_pos, sigma)

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

# reconstruct coil views to return result of parallel image
plt.figure()
plt.suptitle('Sum-of-squares Coil view reconstruction', fontsize=16)
parallel_image = Recon_functions.sum_of_squares(n_coils,np.fft.fftshift(np.fft.fft2(coil_view)))
plt.imshow(abs(parallel_image))

## K-Sapce transformations

# fourier transform image and sensitivities to get K-Space raw data
plt.figure()
plt.suptitle('Coil view (with noise)', fontsize=16)
img_kspace = np.fft.fftshift(np.fft.fft2(img_np))
coil_view_kspace = []
for index,i in enumerate(coil_view):
    x = np.fft.fftshift(np.fft.fft2(i))
    sigma_noise = 0 # 0-15: Low, 15-30: Medium, 30>: High
    noise = np.random.normal(0, sigma_noise, x.shape)
    noise_im = np.random.normal(0, sigma_noise, x.shape) * 1j
    # adding noise to K-Space
    noise = noise + noise_im
    x += noise
    coil_view_kspace.append(x)
    plt.subplot(4,4,index+1)
    #plt.imshow(abs(np.log(x)))
    plt.imshow(abs(np.fft.ifft2(np.fft.ifftshift(x))))
    plt.colorbar()
coil_view_kspace = np.array(coil_view_kspace,dtype=complex)

print(len(coil_view_kspace[0]))

## Parallel imaging: Sampling K-Space coil views with ACS

# acceleration rate (number of phase encoding lines sampled)
R = 2

# undersampling K-Space and getting ACS data
sampled_coils_ks, coil_under_sampled, coil_ACS, coil_ACS_zeros, ACS_row_min, ACS_row_max = Recon_functions.undersampling(coil_view_kspace, R)
# defining the kernel size for the GRAPPA reconstruction
kernel_size = [2, 3]

plt.figure()
plt.suptitle('undersampled coils', fontsize=16)
for i,j in enumerate(coil_under_sampled):
    plt.subplot(4,4,i+1)
    plt.imshow(abs(np.fft.ifft2(np.fft.ifftshift(j))))
    plt.colorbar()

## perform GRAPPA reconstruction
g2 = GRAPPA.GRAPPA(kernel_size, R, coil_ACS, coil_under_sampled)

# show the final GRAPPA reconstructed image
plt.figure()
plt.suptitle('Sum-of-square full GRAPPA reconstruction', fontsize=16)
plt.imshow(abs(g2.image))

plt.figure()
plt.suptitle('coil GRAPPA reconstruction', fontsize=16)
for i in range(n_coils):
    plt.subplot(4, 4, i + 1)
    plt.imshow(abs(np.log(g2.kspace[i])))

plt.figure()
plt.suptitle('ACS GRAPPA reconstruction', fontsize=16)
ACS_interpolated = []
for i in range(n_coils):
    x = np.zeros_like(sampled_coils_ks[0], dtype=complex)
    x[ACS_row_min:ACS_row_max+1] = g2.ACS_data[i,:,:]
    ACS_interpolated.append(x)
ACS_interpolated = np.array(ACS_interpolated)
ACS_GRAPPA_image = Recon_functions.sum_of_squares(n_coils,ACS_interpolated)
plt.imshow(abs(ACS_GRAPPA_image))

plt.figure()
plt.suptitle('ACS data', fontsize=16)
ACS_image = Recon_functions.sum_of_squares(n_coils,coil_ACS_zeros)
plt.imshow(abs(ACS_image))

plt.show()


