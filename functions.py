import numpy as np
import cv2
import math

##########################################################################
############################## common ####################################

# Global vars for use in this file
global_rain_direction = 0
rainy_patches = list()
patchwidth = 0

# Returns frobenius norm of an image
def frobenius_norm(I):
    return np.linalg.norm(I, ord=None, axis=None, keepdims=False)

# Normalize vector
def normalize(v):
    return v / np.sqrt(np.sum(v ** 2))

# Convert the image into the range of [0.0, 1.0]
def img_normalize(img):
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    output = (img.astype('float')-min_val)/(max_val - min_val)
    return output

# Gets x and y derivative images
def partial_both_grayscale(img):
    n_row = img.shape[0]
    n_col = img.shape[1]
    gx = np.zeros((n_row-2, n_col-2), float)
    gy = np.zeros((n_row-2, n_col-2), float)
    for i in range(1, n_row-2):
        for j in range(1, n_col-2):
            gx[i-1, j-1] = (img[i, j+1]-img[i, j-1])
            gy[i-1, j-1] = (img[i+1, j]-img[i-1, j])
    return gx, gy

# Calculates the gradient angle at each pixel given both derivatives
def grad_angle(gx, gy):
    n_row = gx.shape[0]
    n_col = gx.shape[1]
    angles = np.zeros((n_row, n_col), float)
    for i in range(1, n_row):
        for j in range(1, n_col):
            y = gy[i-1, j-1]
            x = gx[i-1, j-1]
            angles[i-1, j-1] = math.atan2(y, x)
    return angles

# Partial derivative of R in x direction
def partial_x(R):
    rows = R.shape[0]
    cols = R.shape[1]
    gx = np.zeros((rows - 2, cols - 2, 3), float)
    for i in range(1, rows - 2):
        for j in range(1, cols - 2):
            gx[i - 1, j - 1] = (R[i, j + 1] - R[i, j - 1]) / 2.0
    return gx

# Partial derivative of R in y direction
def partial_y(R):
    rows = R.shape[0]
    cols = R.shape[1]
    gy = np.zeros((rows - 2, cols - 2, 3), float)
    for i in range(1, rows - 2):
        for j in range(1, cols - 2):
            gy[i - 1, j - 1] = (R[i + 1, j] - R[i - 1, j]) / 2.0
    return gy

# Returns the gradient at a specific pixel
def pixel_gradient(I, x, y):
    vector = []
    partx = (I[x, y + 1] - I[x, y - 1]) / 2.0
    party = (I[x + 1, y] - I[x - 1, y]) / 2.0
    vector.append((partx[0] + partx[1] + partx[2]) / 3)
    vector.append((party[0] + party[1] + party[2]) / 3)
    return [vector[0], vector[1]]

# Return starting x and y value of patch centered at pixel value
def get_patch(pixel):
    startx = int(max(0, pixel[0] - patchwidth/2))
    starty = int(max(0, pixel[1] - patchwidth/2))
    return (startx, starty)

# Based on starting pixels of two patches, slices the image and returns the difference of the two
def patch_difference(I, patch1, patch2):
    patch1_img = I[patch1[1]:patch1[1] + patchwidth, patch1[0]:patch1[0] + patchwidth]
    patch2_img = I[patch2[1]:patch2[1] + patchwidth, patch2[0]:patch2[0] + patchwidth]
    return patch1_img - patch2_img


### All of the following functions are attempts to recreate the functions defined in the paper

##########################################################################
############################### psi ######################################

# psi = sparsity prior
# This effectively removes rain streaks, but background details as well
gamma = 5  # some weight
M = 20

def regularize_psi(B):
    alphavector = normalize(alpha(B))
    rows = B.shape[0]
    cols = B.shape[1]
    # loop through each pixel and add to sum
    sum = 0
    for i in range(rows - 1):
        for j in range(cols - 1):
            pixel = B[i, j]
            sum += normalize(alpha_i(B, pixel) - mu_i(B, pixel))
    return alphavector + gamma * sum

## TODO: All of these

# return vector of all alpha_i
def alpha(B):
    vector = []
    rows = B.shape[0]
    cols = B.shape[1]
    for i in range(rows - 1):
        for j in range(cols - 1):
            pixel = B[i, j]
            vector.append(alpha_i(B, pixel))
    return vector

# alpha_i = sparse code of patch centered at pixel i
def alpha_i(B, pixel):
    return 0

# alpha_i_m = sparse code of mth similar patch
def alpha_i_m(B, pixel, m):
    return 0

# mu_i = weighted average of the sparse codes of the M nonlocal patches that are
# the most similar to the patch centered at pixel i
def mu_i(B, pixel):
    return 0

# tau_i_m = distance between a_i_m and patch centered at pixel i
def tau_i_m(B, pixel, m):
    return 0


##########################################################################
############################### phi ######################################

# phi = Rain direction prior
# Helps put back in background details that could be mistaken for streaks

epsilon_1 = 0.0001 # Constant to avoid division by 0

def regularize_phi(B):
    rows = B.shape[0]
    cols = B.shape[1]
    # loop through each pixel and add to sum
    sum = 0
    for i in range(rows - 1):
        for j in range(cols - 1):
            term = theta_i(B, i, j) + epsilon_1
            sum += (1 / term)
    return sum

# compares rain direction and pixel gradient
def theta_i(B, x, y):
    gradient = pixel_gradient(B, x, y)
    numerator = np.dot(global_rain_direction, gradient)
    denominator = np.linalg.norm(gradient) + epsilon_1
    return abs(numerator) / denominator


##########################################################################
############################## omega #####################################

# omega = rain layer prior
# Push scene details from R back into B

eta = 1.2 # sensitivity parameter

def regularize_omega(R, I):
    rows = R.shape[0]
    cols = R.shape[1]
    # Calculate derivative of R
    R_gray = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    R_gray_norm = img_normalize(R_gray)
    dx, dy = partial_both_grayscale(R_gray_norm)
    # loop through each pixel and add to sum
    sum = 0
    for i in range(rows - 2):
        for j in range(cols - 2):
            pixel = (i, j)
            dx_pixel = dx[pixel]
            dy_pixel = dy[pixel]
            term1 = weight_x(I, pixel, dx_pixel) * (dx_pixel ** 2)
            term2 = weight_y(I, pixel, dy_pixel) * (dy_pixel ** 2)
            sum += (term1 + term2)
    return sum

# Smoothing weight on pixel i in x direction
def weight_x(I, pixel, partial):
    term = partial * gamma_i(I, pixel)
    return abs(term) ** eta

# Smoothing weight on pixel i in y direction
def weight_y(I, pixel, partial):
    term = partial * gamma_i(I, pixel)
    return abs(term) ** eta

# Similarity map
def gamma_i(I, pixel):
    vector = list()
    patch = get_patch(pixel)
    # Go through each rain patch and compare similarity, return min of norm of similarities
    for rainy_patch in rainy_patches:
        difference = patch_difference(I, patch, rainy_patch)
        vector.append(frobenius_norm(difference))
    return min(vector)

