# Python file containing all functions relating to reconstruction for parallel imaging

# import all relvant modules
import numpy as np
import matplotlib.pyplot as plt


# function: sum of squares reconstruction
def sum_of_squares(n_coils, coil_kspace):
    # initalising image for iteration
    image_total = np.zeros_like(coil_kspace[0], dtype=float)
    # for loop to reconstruct the full GRAPPA image (sum of squares)
    for i in range(n_coils):
        # squeeze into 1 dimension
        k_recon = np.squeeze(coil_kspace[i, :, :])
        # iteratively improve the image with the square of the new coil information
        image_total += np.abs((np.fft.ifft2(np.fft.ifftshift(k_recon)))) ** 2

    # sqrt the final iteration to retrieve the final image
    final_image = image_total ** 0.5
    return final_image