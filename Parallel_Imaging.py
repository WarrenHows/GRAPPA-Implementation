# import all nessersary modules/libraries
import imageio.v2 as iio
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom  # import test image
import numpy as np
from scipy import fftpack
from scipy import signal
import Recon_functions
import GRAPPA
import time


# GRAPPA pre-processing starts
pre_start = time.perf_counter()

# import test image, convert to numpy array and find dimensions
#img = iio.imread("childMRI.png")
img = iio.imread("T1.png")
#img = iio.imread("T2.png")
#img = shepp_logan_phantom()
img_np = np.array(img)
img_np = img_np / img_np.max()  # normalise to 0-1 range
img_np = img_np
Ny,Nx = np.shape(np.array(img))
noise = 1

plt.figure()
plt.imshow(img_np, cmap='gray')

## Coil Implementation

# coil parameters
n_coils = 32
sigma = 30
# define the desired SNR
SNR = 30

# creating coils
coil_sensitivities, coil_view = Recon_functions.creating_coils(img_np, Nx, Ny, n_coils, sigma)

# displaying the coil sensitivities
Recon_functions.display_images_(coil_sensitivities, 'Coil sensitivities')

# reconstruct coil views to return result of parallel image
coil_view_kspace = np.array([np.fft.fftshift(np.fft.fft2(coil_view[i])) for i in range(n_coils)])
plt.figure()
plt.suptitle('Sum-of-squares Coil view reconstruction', fontsize=16)
parallel_image = Recon_functions.sum_of_squares(n_coils, coil_view_kspace)
plt.imshow(abs(parallel_image), cmap='grey')
plt.axis('off')

# k-space transformations

# apply noise to image and calculate SNR of noisy sum of squares ground truth
if noise == 1:
    coil_view_kspace = Recon_functions.applying_noise_SNR(coil_view, SNR)
    noisy_gt_image = Recon_functions.sum_of_squares(n_coils, coil_view_kspace)
else:
    coil_view_kspace = np.fft.fft2(coil_view)



# display the noisy image
Recon_functions.display_images_(np.array([np.abs(np.fft.ifft2(np.fft.ifftshift((i)))) for i in coil_view_kspace]), 'Coil view (with noise)')

# confirmation print
print(len(coil_view_kspace[0]))

## Parallel imaging: Sampling K-Space coil views with ACS

# acceleration rate (number of phase encoding lines sampled)
R = 2

# undersampling K-Space and getting ACS data
sampled_coils_ks, coil_under_sampled, coil_ACS, coil_ACS_zeros, ACS_row_min, ACS_row_max = Recon_functions.undersampling(coil_view_kspace, R)
# defining the kernel size for the GRAPPA reconstruction
kernel_size = [4, 5]

# display the undersampled coils
Recon_functions.display_images_(np.array([np.abs(np.fft.ifft2(np.fft.ifftshift((i)))) for i in coil_under_sampled]), 'undersampled coils')
undersampled_image = Recon_functions.sum_of_squares(n_coils, coil_under_sampled)
plt.figure()
plt.imshow(abs(undersampled_image), cmap='gray')
plt.axis('off')

pre_end = time.perf_counter()
# GRAPPA pre-processing ends
print(f"Pre-processing time:       {pre_end - pre_start:.4f} seconds")

# perform strong GRAPPA reconstruction
strong_start = time.perf_counter()
sg = GRAPPA.GRAPPA(kernel_size, R, coil_ACS, coil_under_sampled)
grappa_image = sg.image
grappa_kspace = sg.kspace
strong_end = time.perf_counter()
print(f"GRAPPA time:        {strong_end - strong_start:.4f} seconds")

abs_diff = grappa_image - parallel_image
plt.figure()
plt.subplot(1,3,1)
plt.imshow(np.abs(grappa_image), cmap='gray')
plt.title(f'GRAPPA reconstruction image (R={R})')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(np.abs(parallel_image), cmap='gray')
plt.axis('off')
plt.title('Ground truth image')
plt.subplot(1,3,3)
plt.imshow(np.abs(abs_diff))
plt.title(f'Absolute difference (R={R})')
plt.axis('off')

plt.show()