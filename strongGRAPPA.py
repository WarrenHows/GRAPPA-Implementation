## strongGRAPPA class for GRAPPA reconstruction implementation using entirety of ACS for weight calibration

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

        # calculate each of the R-1 kernels individually
        for kernel_num in total_kernel_num:
            # calculate relative indices for the source positions in kernel
            kernel_rel_rows, kernel_rel_cols = self.relative_indices(R, kernel_size, kernel_num)

            # finding the actual target indices across the ACS
            pad_ACS_trg_rows, pad_ACS_trg_cols, num_ACS_trg = self.ACS_trg_indices_calc(padded_ACS_data)

            # finding the actual source values across the ACS
            S_ACS = self.src_val_calc(coils, kernel_size, num_ACS_trg, kernel_rel_rows, kernel_rel_cols,
                                      pad_ACS_trg_rows, pad_ACS_trg_cols, padded_ACS_data)

            # calculating the ACS weights
            w = self.weight_calc(coils, num_ACS_trg, S_ACS, pad_ACS_trg_rows, pad_ACS_trg_cols, padded_ACS_data)

            # finding the actual target indices across all of K-Space
            pad_trg_rows, pad_trg_cols, trg_rows, trg_cols, num_trg = self.trg_indices_calc(kspace, R, kernel_size, n_rows,
                                                                                            n_cols, kernel_num,
                                                                                            pad_size_y, pad_size_x)

            # finding actual source values across all of K-Space
            S = self.src_val_calc(coils, kernel_size, num_trg, kernel_rel_rows, kernel_rel_cols, pad_trg_rows,
                                  pad_trg_cols, padded_kspace)

            # calculating trg values
            M = np.matmul(w, S)

            # repopulating K-Space with calculated targets
            kspace = self.apply_targets(coils, pad_trg_rows, pad_trg_cols, M, padded_kspace, kspace, pad_size_y,
                                        pad_size_x, n_rows, n_cols)

        # assemble all of the coil views into an image
        self.kspace = kspace
        self.ACS_data = ACS_data
        image = Recon_functions.sum_of_squares(coils, self.kspace)
        self.image = image

    def relative_indices(self, R, kernel_size, kernel_num):

        # calculate distance kernel is operating over
        kernel_dist_y = R * kernel_size[0]
        kernel_dist_x = 1 * kernel_size[1]

        # find indices of source points within the area of 1 kernel
        mask_src = np.zeros((kernel_dist_y, kernel_dist_x), dtype=bool)
        mask_src[:kernel_dist_y:R, :kernel_dist_x:1] = True
        kernel_src_rows, kernel_src_cols = np.where(mask_src == True)

        # find indices of target points within the area of 1 kernel
        mask_1trg = np.zeros((kernel_dist_y, kernel_dist_x), dtype=bool)
        rw_index = int(kernel_dist_y / 2 - R + kernel_num)
        col_index = int((kernel_dist_x + 1) / 2 - 1)
        mask_1trg[rw_index, col_index] = True
        kernel_trg_rows, kernel_trg_cols = np.where(mask_1trg == True)

        # calculating relative indices of source points in reference to the target point in small kernel above (this is applicable to all points for reconstruction)
        kernel_rel_rows = kernel_src_rows - kernel_trg_rows
        kernel_rel_cols = kernel_src_cols - kernel_trg_cols

        return kernel_rel_rows, kernel_rel_cols

    def ACS_trg_indices_calc(self, padded_ACS_data):

        # choose every ACS position except for the padding
        pad_trg_row, pad_trg_col = np.where(padded_ACS_data[0] != 0)

        # find number of targets
        num_trg = len(pad_trg_row)

        return pad_trg_row, pad_trg_col, num_trg

    def src_val_calc(self, coils, kernel_size, num_trg, kernel_rel_rows, kernel_rel_cols, pad_trg_row, pad_trg_col, data):

        # adding every relative source index by every target index to find the exact source indices using loops
        src_rows = np.array([rw + kernel_rel_rows for rw in pad_trg_row], dtype=int)
        src_cols = np.array([col + kernel_rel_cols for col in pad_trg_col], dtype=int)

        # calculating the dimensions of the S matrix (source matrix for weight calculation)
        num_kernel_vals = kernel_size[0] * kernel_size[1]
        total_num_src_values = coils * num_kernel_vals
        # initalise the S matrix with dimensions
        S = np.zeros((total_num_src_values, num_trg), dtype=complex)

        # loop to fill up the matrix of all source values in the sample (assembling S)
        # iterate over all of the rows and columns which act as valid sources
        for index, (rw, col) in enumerate(zip(src_rows, src_cols)):
            j = np.arange(len(rw)) # list of length of sources
            # iterate over all coils
            for coil in np.arange(coils):
                S[j,index] = data[coil,rw,col] # S rows correspond to all positions, S cols correspond to all sources at position
                j += len(rw) # increase the length of j for next iteration of coils

        return S

    def weight_calc(self, coils, num_trg, S_ACS, pad_trg_rows, pad_trg_cols, padded_ACS_data):

        # creating a matrix of zeros with dims of number of coils X number of targets to contain all target values (initalising M matrix)
        M_ACS = np.zeros((coils, num_trg), dtype=complex)

         # define dimensions of ACS
        coils, Ny_ACS, Nx_ACS = padded_ACS_data.shape
        coils = np.arange(coils)

        # iterate over all of the ACS data (as all ACS can act as a target) to create target matrix
        for M_col, (rw, col) in enumerate(zip(pad_trg_rows,pad_trg_cols)):
            # broadcast over all weight matrix rows and fill up, weight column with ACS data at each location
            M_ACS[:,M_col] = padded_ACS_data[coils, rw, col]

        # calculating weights using Linear Least Squares
        # current format of M = W * S is not compatible therefore rewrite as M.T = S.T * W.T
        W_T, residuals, rank, s = np.linalg.lstsq(S_ACS.T, M_ACS.T, rcond=None)
        # transpose the weights again to get them in the right dimensions
        w = W_T.T

        return w

    def trg_indices_calc(self, kspace ,R, kernel_size, n_rows, n_cols, kernel_num, pad_size_y, pad_size_x):

        # all rows in unpadded space
        all_rows = np.arange(n_rows)

        # find every place where the k-space is empty (has a 0)
        trg_row, trg_col = np.where(kspace[0] == 0)
        prev_num_trg_row = len(trg_row)

        # choose the rows which have a remainder equal to the kernel_num when divided by R (that is the offset we wish to reconstruct)
        trg_row = trg_row[trg_row % R == kernel_num]
        new_num_trg_row = len(trg_row)

        # num target columns must equal number of rows
        new_num_trg_col = new_num_trg_row

        trg_col = trg_col[:new_num_trg_col]

        # pad row index
        pad_trg_row = trg_row + pad_size_y

        # retrieve and pad target columns
        pad_trg_col = trg_col + pad_size_x

        # find number of targets
        num_trg = len(pad_trg_row)

        return pad_trg_row, pad_trg_col, trg_row, trg_col, num_trg

    def apply_targets(self, coils, pad_trg_rows, pad_trg_cols, M, pad_kspace, kspace, pad_size_y, pad_size_x, n_rows, n_cols):

        # iterate across the target values inputting them into padded K-space
        j1 = 0 # counter for the columns of the target array
        for j, (rw, col) in enumerate(zip(pad_trg_rows, pad_trg_cols)):
                pad_kspace[:,rw,col] = M[:,j]
                x = pad_kspace[0]

        # input the new values into the un-padded K-space
        ky_pad_range = slice(pad_size_y,n_rows+pad_size_y)
        ky_range = slice(0,n_rows)
        kx_pad_range = slice(pad_size_x,n_cols+pad_size_x)
        kx_range = slice(0,n_cols)
        kspace[:,ky_range,kx_range] = pad_kspace[:,ky_pad_range,kx_pad_range]
        return kspace
