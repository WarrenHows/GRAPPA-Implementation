## GRAPPA class for GRAPPA reconstruction implementation

# import all nessersary modules/libraries
import imageio.v2 as iio
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom  # import test image
import numpy as np
from scipy import fftpack
from scipy import signal
import Recon_functions

# arguments
# kernel_size = GRAPPA kernel y direction (even) and x direction (odd) lengths
# R = accleration rate
# ACS_data = ACS values needed for calculation of the weights
# kspace = undersampled data needed to be reconstructed

## GRAPPA class for GRAPPA implementation
class GRAPPA:
    def __init__(self, kernel_size, R, ACS_data, kspace):

        # convert to numpy arrays for later computation
        kspace = np.array(kspace,dtype=complex)
        ACS_data = np.array(ACS_data,dtype=complex)

        # find dimensions of kspace and ACS (coils and cols will be the shape for both)
        coils,n_rows,n_cols = np.shape(kspace)
        n_ACS_rows,n_ACS_cols = np.shape(ACS_data[0])

        # number of different kernels used for the GRAPPA equation
        total_kernel_num = np.array(range(1,R))

        # calculate the pad size in each direction for the image
        pad_size_y = int(R * kernel_size[0] / 2)
        pad_size_x = int(kernel_size[1] // 2)

        # pad kspace and ACS data
        padded_kspace = []
        padded_ACS_data = []
        for coil in range(coils):
            padded_kspace.append(np.pad(kspace[coil],pad_width=((pad_size_y,pad_size_y),(pad_size_x,pad_size_x)),mode='constant', constant_values=0))
            padded_ACS_data.append(np.pad(ACS_data[coil],pad_width=((pad_size_y,pad_size_y),(pad_size_x,pad_size_x)),mode='constant', constant_values=0))
        padded_kspace = np.array(padded_kspace)
        padded_ACS_data = np.array(padded_ACS_data)
        # dimensions of padded data
        coils,n_padded_rows,n_padded_cols = np.shape(padded_kspace)
        coils,n_padded_ACS_rows,n_padded_ACS_cols = np.shape(padded_ACS_data)

        # ranges for kspace
        ky_range = slice(pad_size_y, n_rows - pad_size_y, R)
        kx_range = slice(pad_size_x, n_cols)
        # undersampling ACS data
        ky_ACS_range = slice(pad_size_y, n_ACS_rows+1-pad_size_y, R)
        kx_ACS_range = slice(pad_size_x, n_ACS_cols+1)
        usamp_ACS_data = []
        for coil in range(coils):
            usamp_ACS = np.zeros_like(padded_ACS_data[coil])
            usamp_ACS[ky_ACS_range,kx_ACS_range] = padded_ACS_data[coil, ky_ACS_range, kx_ACS_range]
            usamp_ACS_data.append(usamp_ACS)
        usamp_ACS_data = np.array(usamp_ACS_data)


        # calculate each of the R-1 kernels individually
        for kernel_num in total_kernel_num:

            # calculate relative indices for the source positions in kernel
            kernel_rel_rows, kernel_rel_cols = self.relative_indices(R, kernel_size, n_rows, n_cols, kernel_num)

            # finding the actual target indices across the ACS
            pad_ACS_trg_rows, pad_ACS_trg_cols, ACS_trg_rows, ACS_trg_cols, num_ACS_trg = self.trg_indices_calc(R, kernel_size, n_ACS_rows, n_ACS_cols, kernel_num, pad_size_y, pad_size_x)

            # finding the actual source values across the ACS
            S_ACS = self.src_val_calc(coils, kernel_size, num_ACS_trg, kernel_rel_rows, kernel_rel_cols, pad_ACS_trg_rows, pad_ACS_trg_cols, padded_ACS_data)

            # calculating the ACS weights
            w = self.weight_calc(coils, num_ACS_trg, S_ACS, ACS_trg_rows, ACS_trg_cols, ACS_data)

            # interpolating ACS region for clarification
            M_ACS_interpolated = np.matmul(w,S_ACS)

            # calculate the ACS region from GRAPPA reconstruction
            ACS_data = self.apply_targets(coils, pad_ACS_trg_rows, pad_ACS_trg_cols, M_ACS_interpolated, padded_ACS_data, ACS_data, pad_size_y, pad_size_x, n_ACS_rows, n_ACS_cols)

            # finding the actual target indices across all of K-Space
            pad_trg_rows, pad_trg_cols, trg_rows, trg_cols, num_trg = self.trg_indices_calc(R, kernel_size, n_rows, n_cols, kernel_num, pad_size_y, pad_size_x)

            # finding actual source values across all of K-Space
            S = self.src_val_calc(coils, kernel_size, num_trg, kernel_rel_rows, kernel_rel_cols, pad_trg_rows, pad_trg_cols, padded_kspace)

            # calculating trg values
            M = np.matmul(w,S)

            # repopulating K-Space with calculated targets
            kspace = self.apply_targets(coils, pad_trg_rows, pad_trg_cols, M, padded_kspace, kspace, pad_size_y, pad_size_x, n_rows, n_cols)


        # assemble all of the coil views into an image
        self.kspace = kspace
        self.ACS_data = ACS_data
        image = Recon_functions.sum_of_squares(coils, self.kspace)
        self.image = image
        ACS_image = Recon_functions.sum_of_squares(coils, self.ACS_data)


    def relative_indices(self, R, kernel_size, n_rows, n_cols, kernel_num):

        # calculate distance kernel is operating over
        kernel_dist_y = R * kernel_size[0]
        kernel_dist_x = 1 * kernel_size[1]

        # find indicies of source points within the area of 1 kernel
        mask_src = np.zeros((n_rows, n_cols), dtype=bool)
        mask_src[:kernel_dist_y:R, :kernel_dist_x:1] = True
        kernel_src_rows, kernel_src_cols = np.where(mask_src == True)
        kernel_src = np.stack((kernel_src_rows, kernel_src_cols))

        # find the index for the 1 desired target point within the area of 1 kernel
        mask_1trg = np.zeros((n_rows, n_cols), dtype=bool)
        rw_index = int(kernel_dist_y / 2 - R + kernel_num)
        col_index = int((kernel_dist_x + 1) / 2 - 1)
        mask_1trg[rw_index, col_index] = True
        kernel_trg_rows, kernel_trg_cols = np.where(mask_1trg == True)
        kernel_trg = np.stack((kernel_trg_rows, kernel_trg_cols))

        # calculating relative indicies of source points in reference to the target point in small kernel above (this is applicable to all points for reconstruction)
        kernel_rel_rows = kernel_src_rows - kernel_trg_rows
        kernel_rel_cols = kernel_src_cols - kernel_trg_cols
        kernel_rel_idx = np.stack((kernel_rel_rows, kernel_rel_cols))
        return kernel_rel_rows, kernel_rel_cols

    def trg_indices_calc(self, R, kernel_size, n_rows, n_cols, kernel_num, pad_size_y, pad_size_x):

        # all rows in unpadded space
        all_rows = np.arange(n_rows)

        # choose the rows which have a remainder equal to the kernel_num when divided by R
        trg_row = all_rows[all_rows % R == kernel_num]

        # pad row index
        pad_trg_row = trg_row + pad_size_y

        # retrieve and pad target columns
        trg_col = np.arange(n_cols)
        pad_trg_col = trg_col + pad_size_x

        # find number of targets
        num_trg = len(pad_trg_row) * len(pad_trg_col)

        return pad_trg_row, pad_trg_col, trg_row, trg_col, num_trg

    def src_val_calc(self, coils, kernel_size, num_trg, kernel_rel_rows, kernel_rel_cols, pad_trg_row, pad_trg_col, data):

        # adding every relative source index by every target index to find the exact source indicies using loops
        src_rows = []
        src_cols = []
        for rw in pad_trg_row:
            src_rows.append(rw + kernel_rel_rows)  # find all source row indices for sample
        for col in pad_trg_col:
            src_cols.append(col + kernel_rel_cols)  # find all source column indices for sample

        # calculating the dimensions of the S matrix (source matrix for weight calculation)
        num_kernel_vals = kernel_size[0] * kernel_size[1]
        total_num_src_values = coils * num_kernel_vals
        # initalise the S matrix with dimensions
        S = np.zeros((total_num_src_values, num_trg), dtype=complex)

        # loop to fill up the matrix of all source values in the sample (assembling S)
        j = 0
        for rw in src_rows:
            for col in src_cols:
                i = 0
                for coil in range(coils):
                    for rw_idx, col_idx in zip(rw,col):
                        S[i, j] = data[coil, rw_idx, col_idx]
                        i += 1
                j += 1

        return S

    def weight_calc(self, coils, num_trg, S_ACS, trg_rows, trg_cols, ACS_data):

        # creating a matrix of zeros with dims of number of coils X number of targets to contain all target values (initalising M matrix)
        M_ACS = np.zeros((coils, num_trg), dtype=complex)

        # finding values of the ACS targets (assembling M matrix)
        i = 0
        for coil in range(coils):
            j = 0
            for rw_idx in trg_rows:
                for col_idx in trg_cols:
                    M_ACS[i, j] = ACS_data[coil, rw_idx, col_idx]
                    j += 1
            i += 1

        # calculating weights using Linear Least Squares
        # current format of M = W * S is not compatible therefore rewrite as M.T = S.T * W.T
        W_T, residuals, rank, s = np.linalg.lstsq(S_ACS.T, M_ACS.T, rcond=None)
        # transpose the weights again to get them in the right dimensions
        w = W_T.T

        return w

    def apply_targets(self, coils, pad_trg_rows, pad_trg_cols, M, pad_kspace, kspace, pad_size_y, pad_size_x, n_rows, n_cols):

        # iterate across the target values inputting them into padded K-space
        for coil in range(coils):
            j = 0
            for row in pad_trg_rows:
                for col in pad_trg_cols:
                    pad_kspace[coil, row, col] = M[coil, j]
                    j += 1
            # input the new values into the un-padded K-space
            ky_pad_range = slice(pad_size_y,n_rows+pad_size_y)
            ky_range = slice(0,n_rows)
            kx_pad_range = slice(pad_size_x,n_cols+pad_size_x)
            kx_range = slice(0,n_cols)
            kspace[coil,ky_range,kx_range] = pad_kspace[coil,ky_pad_range,kx_pad_range]
        return kspace
