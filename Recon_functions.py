# Python file containing all functions relating to reconstruction for parallel imaging

# import all relvant modules
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import GRAPPA


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

def sos_func(x):
    # initialise the final image
    total_image = np.zeros_like(x[0], dtype=complex)
    # iterate across all elements of x
    for i in x:
        # add the sum of the square to the final image
        total_image += (np.abs(i) ** 2)
    # perform square root to final sum
    total_image = np.sqrt(total_image)

    return total_image


def undersampling(coil_view_kspace, R, num_ACS = 40):
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

# function for displaying images
def display_images_(image_batch, title=''):

    # find the number of images within the batch
    length = image_batch.shape[0]

    # number of images in batch should be divisble by 4
    rw = int(length / 4)

    plt.figure()
    plt.suptitle(title, fontsize=16)
    for i in range(length):
        plt.subplot(rw, 4, i + 1)
        plt.imshow(np.abs(image_batch[i]), cmap='gray')
        plt.axis('off')

def creating_coils(img_np, Nx, Ny, n_coils, sigma, noise=0):

    # define locations of all coils
    if isinstance(n_coils ** 0.5, int):
        # identifying the positions of the coils on the image
        coil_x_pos = np.linspace(0, int(Nx), int(n_coils ** 0.5), endpoint=False)
        coil_y_pos = np.linspace(0, int(Ny), int(n_coils ** 0.5), endpoint=False)

    else:
        # identifying the positions of the coils on the image
        coil_x_pos = np.linspace(int(Nx / 4), int(Nx * 0.75), int(4))
        coil_y_pos = np.linspace(int(Ny / 4), int(Ny * 0.75), int(n_coils / 4))

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

    # multiply the image by the sensitivities
    coil_view = []
    # iterate over the coil sensitivities to multiply the k-space to get the coil views
    for i in enumerate(coil_sensitivities):
        # each individual coil view
        x = np.multiply(i[1], img_np)
        coil_view.append(x)
    coil_view = np.array(coil_view, dtype=complex)

    return coil_sensitivities, coil_view


def applying_noise_SNR(coil_view, SNR, noise=1):

    # initialise empty array
    coil_view_kspace = []
    # iterate over all of the coil views
    for index, i in enumerate(coil_view):
        # fourier transform the individual views of the coil
        x = np.fft.fftshift(np.fft.fft2(i))
        # signal power for SNR calculation
        signal_power = np.mean(np.abs(x) ** 2)
        # define the noise
        sigma_noise = 1
        noise_real = np.random.normal(0, sigma_noise, x.shape)
        # define the imaginary noise component
        noise_im = np.random.normal(0, sigma_noise, x.shape) * 1j
        # creating complex noise
        noise_total = noise_real + noise_im
        # calculate power for SNR calculation
        noise_power = np.mean(np.abs(noise_total) ** 2)
        SNR_c = signal_power / noise_power
        # adjust to be desired SNR
        noise_total *= np.sqrt(SNR_c / SNR)
        # adding noise to kspace
        x += noise_total
        coil_view_kspace.append(x)
    coil_view_kspace = np.array(coil_view_kspace, dtype=complex)

    return coil_view_kspace


# Helper: resolve subplot grid from coil count
# ─────────────────────────────────────────────────────────────────────────────

def _coil_grid(n_coils):
    """Return (n_rows, n_cols=4) for coil subplot grids."""
    mapping = {8: 2, 16: 4, 32: 8, 64: 16}
    if n_coils not in mapping:
        raise ValueError(f"Unsupported n_coils={n_coils}. Expected one of {list(mapping.keys())}.")
    return mapping[n_coils], 4


# ─────────────────────────────────────────────────────────────────────────────
# 1. Setup images
#    - coil sensitivity maps
#    - fully-sampled coil views (image domain) + SoS
#    - undersampled coil views + SoS
#    - ACS coil views (embedded in zeros) + SoS
# ─────────────────────────────────────────────────────────────────────────────

def plot_setup_images(n_coils, R, SNR,
                      coil_sensitivities,
                      coil_view_kspace,
                      coil_under_sampled,
                      coil_ACS_zeros,
                      parallel_image):
    """
    Plots acquisition / setup images:
      1. Coil sensitivity maps
      2. Fully-sampled coil views (image domain) + SoS
      3. Undersampled coil views (image domain) + SoS
      4. ACS coil views (image domain) + SoS
    """
    rw, rc = _coil_grid(n_coils)
    tag = f"R={R}, SNR={SNR}, coils={n_coils}"

    # ── 1. Coil sensitivity maps ──────────────────────────────────────────────
    fig, axes = plt.subplots(rw, rc, figsize=(rc * 2.5, rw * 2.5))
    fig.suptitle(f"Coil Sensitivity Maps ({tag})")
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.abs(coil_sensitivities[i]), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Coil {i + 1}", fontsize=7)
    plt.tight_layout()

    # ── 2. Fully-sampled coil views ───────────────────────────────────────────
    fully_samp_views = np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(coil_view_kspace[i])))
                                 for i in range(n_coils)])
    fig, axes = plt.subplots(rw, rc, figsize=(rc * 2.5, rw * 2.5))
    fig.suptitle(f"Fully-Sampled Coil Views ({tag})")
    for idx, ax in enumerate(axes.flat):
        ax.imshow(fully_samp_views[idx], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Coil {idx + 1}", fontsize=7)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.abs(parallel_image), cmap="gray")
    ax.set_title(f"SoS – Fully-Sampled ({tag})")
    ax.axis("off")
    plt.tight_layout()

    # ── 3. Undersampled coil views ────────────────────────────────────────────
    u_views = np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(coil_under_sampled[i])))
                        for i in range(n_coils)])
    fig, axes = plt.subplots(rw, rc, figsize=(rc * 2.5, rw * 2.5))
    fig.suptitle(f"Undersampled Coil Views ({tag})")
    for idx, ax in enumerate(axes.flat):
        ax.imshow(u_views[idx], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Coil {idx + 1}", fontsize=7)
    plt.tight_layout()

    undersampled_sos = sum_of_squares(n_coils, coil_under_sampled)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.abs(undersampled_sos), cmap="gray")
    ax.set_title(f"SoS – Undersampled ({tag})")
    ax.axis("off")
    plt.tight_layout()

    # ── 4. ACS coil views ─────────────────────────────────────────────────────
    acs_views = np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(coil_ACS_zeros[i])))
                          for i in range(n_coils)])
    fig, axes = plt.subplots(rw, rc, figsize=(rc * 2.5, rw * 2.5))
    fig.suptitle(f"ACS Coil Views ({tag})")
    for idx, ax in enumerate(axes.flat):
        ax.imshow(acs_views[idx], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Coil {idx + 1}", fontsize=7)
    plt.tight_layout()

    acs_sos = sum_of_squares(n_coils, coil_ACS_zeros)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.abs(acs_sos), cmap="gray")
    ax.set_title(f"SoS – ACS ({tag})")
    ax.axis("off")
    plt.tight_layout()



# ─────────────────────────────────────────────────────────────────────────────
# 2. Reconstruction images
#    - GRAPPA-reconstructed coil views + SoS
#    - side-by-side reconstruction vs ground truth
#    - absolute difference map
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstruction_images(n_coils, R, SNR,
                               grappa_kspace,
                               grappa_image,
                               parallel_image):
    """
    Plots GRAPPA reconstruction output images:
      1. GRAPPA-reconstructed coil views (image domain) + SoS
      2. Reconstruction vs Ground Truth SoS (side-by-side)
      3. Absolute difference map
    """
    rw, rc = _coil_grid(n_coils)
    tag = f"R={R}, SNR={SNR}, coils={n_coils}"

    # ── 1. GRAPPA-reconstructed coil views ───────────────────────────────────
    recon_views = np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(grappa_kspace[i])))
                            for i in range(n_coils)])
    fig, axes = plt.subplots(rw, rc, figsize=(rc * 2.5, rw * 2.5))
    fig.suptitle(f"GRAPPA Reconstructed Coil Views ({tag})")
    for idx, ax in enumerate(axes.flat):
        ax.imshow(recon_views[idx], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Coil {idx + 1}", fontsize=7)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.abs(grappa_image), cmap="gray")
    ax.set_title(f"SoS – GRAPPA Reconstruction ({tag})")
    ax.axis("off")
    plt.tight_layout()

    # ── 2. Reconstruction vs Ground Truth ─────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Reconstruction vs Ground Truth ({tag})")
    ax1.imshow(np.abs(grappa_image), cmap="gray")
    ax1.set_title(f"GRAPPA Reconstruction (R={R})")
    ax1.axis("off")
    ax2.imshow(np.abs(parallel_image), cmap="gray")
    ax2.set_title("Ground Truth SoS")
    ax2.axis("off")
    plt.tight_layout()

    # ── 3. Absolute difference map ────────────────────────────────────────────
    abs_diff = np.abs(grappa_image - parallel_image)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(abs_diff)
    ax.set_title(f"Absolute Difference ({tag})")
    ax.axis("off")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reconstruction evaluation images
#    - pixel-wise noise std (from noise propagation)
#    - std and mean of repeated reconstructions
# ─────────────────────────────────────────────────────────────────────────────


def plot_reconstruction_evaluation(R, SNR, n_coils,
                                   noise_pixel_wise_std,
                                   all_noise_perfect_images,
                                   std_img,
                                   mean_img,
                                   eff_g_factor_map,
                                   std_img_perfect,
                                   mean_img_perfect):
    """
    Plots quantitative reconstruction evaluation images:
      1. Pixel-wise noise standard deviation (noise propagation) alongside
         the std of the un-reconstructed (R=1) noise — mirrors RAKI's noise panel
      2. Standard deviation and mean of repeated reconstructions (R>1)
      3. Effective g-factor map
      4. R=1 (ground-truth) reconstruction standard deviation and mean
    """
    tag = f"R={R}, SNR={SNR}, coils={n_coils}"

    # ── 1. Pixel-wise noise std + perfect noise std ───────────────────────────
    noise_perfect_std = np.std(all_noise_perfect_images, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Noise Propagation ({tag})")
    im1 = ax1.imshow(np.abs(noise_pixel_wise_std))
    ax1.set_title(f"Pixel-wise Noise Std (R={R})")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(np.abs(noise_perfect_std))
    ax2.set_title("Pixel-wise Noise Std (R=1)")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()

    # ── 2. Std and mean of repeated reconstructions (R>1) ────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Reconstruction Statistics ({tag})")
    im1 = ax1.imshow(np.abs(std_img))
    ax1.set_title(f"Standard Deviation (R={R})")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(np.abs(mean_img))
    ax2.set_title(f"Mean Image (R={R})")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()

    # ── 3. Effective g-factor map ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(eff_g_factor_map)
    ax.set_title(f"Effective g-factor Map ({tag})")
    ax.axis("off")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    # ── 4. Ground-truth (R=1) reconstruction std and mean ────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Ground Truth Reconstruction Statistics (R=1) ({tag})")
    im1 = ax1.imshow(np.abs(std_img_perfect))
    ax1.set_title("Standard Deviation (R=1)")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(np.abs(mean_img_perfect))
    ax2.set_title("Mean Image (R=1)")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save GRAPPA outputs to a single .npz file
# ─────────────────────────────────────────────────────────────────────────────

def save_GRAPPA_outputs(R,
                        grappa_kspace,
                        grappa_image,
                        parallel_image,
                        pixel_wise_std,
                        all_noise_images,
                        all_noise_perfect_images,
                        all_images,
                        all_images_perfect,
                        std_image,
                        mean_image,
                        std_img_perfect,
                        mean_img_perfect,
                        eff_g_factor_map):
    """
    Saves key GRAPPA output arrays to a single compressed NumPy archive.

    File name: GRAPPA_outputs_R_<R>.npz

    Saved arrays
    ─────────────
    grappa_kspace             : complex128 – GRAPPA-interpolated k-space (all coils)
    grappa_image              : float64   – GRAPPA SoS reconstruction
    parallel_image            : float64   – fully-sampled ground truth SoS image
    pixel_wise_std            : float64   – pixel-wise noise std (noise propagation)
    all_noise_images          : float64   – stack of 1000 reconstructed noise images
                                            shape (1000, Ny, Nx)
    all_noise_perfect_images  : float64   – stack of 1000 un-reconstructed (R=1) noise images
                                            shape (1000, Ny, Nx)
    all_images                : float64   – stack of 1000 noisy undersampled reconstructions
                                            shape (1000, Ny, Nx)
    all_images_perfect        : float64   – stack of 1000 fully-sampled noisy reconstructions
                                            shape (1000, Ny, Nx)
    std_image                 : float64   – pixel-wise std across repeated reconstructions
    mean_image                : float64   – mean image across repeated reconstructions
    std_img_perfect           : float64   – pixel-wise std across perfect reconstructions
    mean_img_perfect          : float64   – mean image across perfect reconstructions
    eff_g_factor_map          : float64   – effective g-factor map (std * sqrt(R) / std_perfect)

    Load later with:
        data = np.load("GRAPPA_outputs_R_<R>.npz", allow_pickle=False)
        grappa_kspace            = data["grappa_kspace"]
        grappa_image             = data["grappa_image"]
        parallel_image           = data["parallel_image"]
        pixel_wise_std           = data["pixel_wise_std"]
        all_noise_images         = data["all_noise_images"]
        all_noise_perfect_images = data["all_noise_perfect_images"]
        all_images               = data["all_images"]
        all_images_perfect       = data["all_images_perfect"]
        std_image                = data["std_image"]
        mean_image               = data["mean_image"]
        std_img_perfect          = data["std_img_perfect"]
        mean_img_perfect         = data["mean_img_perfect"]
        eff_g_factor_map         = data["eff_g_factor_map"]
    """
    filename = f"GRAPPA_outputs_R_{R}.npz"
    np.savez_compressed(
        filename,
        grappa_kspace=grappa_kspace,
        grappa_image=grappa_image,
        parallel_image=parallel_image,
        pixel_wise_std=pixel_wise_std,
        all_noise_images=all_noise_images,
        all_noise_perfect_images=all_noise_perfect_images,
        all_images=all_images,
        all_images_perfect=all_images_perfect,
        std_image=std_image,
        mean_image=mean_image,
        std_img_perfect=std_img_perfect,
        mean_img_perfect=mean_img_perfect,
        eff_g_factor_map=eff_g_factor_map,
    )
    print(f"GRAPPA outputs saved to: {filename}")
        
