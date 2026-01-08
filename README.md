# GRAPPA-Implementation
Python implementation of the MRI Reconstruction Algorithm GRAPPA

Implementation has been inspired by Mark Chiew's MATLAB GRAPPA tutorial https://github.com/mchiew/grappa-tutorial

## Parallel_Imaging.py
Script which takes an image, in the script the Shepp-logan Phantom is used and a neonatal brain is commented, and multiplies image by a gaussian sensativity maps to mirror Parrallel Imaging acquisition. These parallel images are then undersampled by an acceleration factor, R, and ACS data is also sampled from each coil. The GRAPPA reconstruction algorithm is called in this script to reconstruct undersampled data.

## Recon_functions.py
Script containing functions which are used in the Parallel_Imaging.py file for undersampling and for reconstruction.
- function 1: sum_of_squares(): for sum of squares reconstruction of the parallel imaging coils
- function 2: undersampleing(): for undersampling the orignal image with acceleration factor, R, and ACS data
- function 3: creating_coils(): for creating Gaussian distributions at certain locations for parallel acquisition

## GRAPPA.py 
Class which contains the GRAPPA reconstruction algorithm. 
- function 1: relative_indices() used to calcualte the indices of the GRAPPA source and targets of the kernel
- function 2: trg_indices_calc() calculates the targets in the image input
- function 3: src_val_calc() calcualtes the source values from the image input
- function 4: weight_calc() calculates the weights of the GRAPPA kernel
- function 5: apply_targets() reinserting the calculated target values into the undersampled image 
