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

def undersampling(coil_view_kspace, R, num_ACS = 24):
    # initalise sampling of image
    coil_ACS = []
    coil_under_sampled = []
    coil_ACS_zeros = []
    sampled_coils_ks = []

    # find centre row based on odd or even size
    if len(coil_view_kspace[0]) % 2 == 0:
        row = int(len(coil_view_kspace[0]) / 2) - 1
    else:
        row = int(len(coil_view_kspace[0]) // 2)

    # ACS range
    abv = int(num_ACS / 2)
    bel = int(num_ACS / 2) - 1
    ACS_row_min = row - abv
    ACS_row_max = row + bel

    # sampling K-Space for each coil (with ACS)
    for i in coil_view_kspace:
        # full K-Space sampling
        x = np.zeros_like(i, dtype=complex)
        y = np.zeros_like(i, dtype=complex)
        z = np.zeros_like(i, dtype=complex)
        x[::R] = i[::R]
        y[::R] = i[::R]
        coil_under_sampled.append(y)
        # including ACS lines with K-Space
        x[ACS_row_min:ACS_row_max + 1:1, :] = i[ACS_row_min:ACS_row_max + 1:1, :]
        z[ACS_row_min:ACS_row_max + 1, :] = i[ACS_row_min:ACS_row_max + 1, :]
        coil_ACS.append(i[ACS_row_min:ACS_row_max + 1:1, :])
        coil_ACS_zeros.append(z)
        sampled_coils_ks.append(x)

    sampled_coils_ks = np.array(sampled_coils_ks)
    coil_under_sampled = np.array(coil_under_sampled)
    coil_ACS_zeros = np.array(coil_ACS_zeros)
    coil_ACS = np.array(coil_ACS)

    return sampled_coils_ks, coil_under_sampled, coil_ACS, coil_ACS_zeros, ACS_row_min, ACS_row_max

def creating_coils(Nx, Ny, coil_y_pos, coil_x_pos, sigma, noise=0):
    # Create spatial grid
    x = np.linspace(0, Nx, Nx)
    y = np.linspace(0, Ny, Ny)
    X, Y = np.meshgrid(x, y)

    # assigning the coordiantes of each coils position
    coil_pos = []
    for i in coil_y_pos:
        for j in coil_x_pos:
            coil_pos.append((i, j))
    # print("coil_pos ",coil_pos)
    coil_pos = np.array(coil_pos)

    # assigning coil sensitivity with Gaussian distribution originating from coil position
    coil_sensitivities = []
    for y0, x0 in coil_pos:
        # calculating the Gaussian distibution at certain coil locations with (imaginary) phase distribution
        G_r = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * (sigma ** 2)))
        G1im = 1j * np.exp(-((X - x0 - 5) ** 2 + (Y - y0 + 10) ** 2) / (2 * (sigma ** 2)))
        coil_sensitivities.append(G_r + G1im)
    coil_sensitivities = np.array(coil_sensitivities, dtype=complex)

    return coil_sensitivities