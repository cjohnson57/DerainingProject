import numpy as np
import cv2
import math

##########################################################################
############################## common ####################################

# Global vars for use in this file
global_rain_direction = 0.0
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
    if max_val == 0 and min_val == 0: # Image is only 0s, just return that
        return img
    output = (img.astype('float')-min_val)/(max_val - min_val)
    return output

# Gets x and y derivative images
def partial_both_grayscale(img):
    rows = img.shape[0]
    cols = img.shape[1]
    gx = np.zeros((rows-2, cols-2), float)
    gy = np.zeros((rows-2, cols-2), float)
    for i in range(1, rows-2):
        for j in range(1, cols-2):
            gx[i-1, j-1] = (img[i, j+1]-img[i, j-1])
            gy[i-1, j-1] = (img[i+1, j]-img[i-1, j])
    return gx, gy

# Calculates the gradient angle at each pixel given both derivatives
def grad_angle(gx, gy):
    rows = gx.shape[0]
    cols = gx.shape[1]
    angles = np.zeros((rows, cols), float)
    for i in range(1, rows):
        for j in range(1, cols):
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
    vector.append(partx / 3)
    vector.append(party / 3)
    return [vector[0], vector[1]]

# Represents the global rain direction as a vector and returns
def global_rain_vector():
    return [1, global_rain_direction]

# Return starting x and y value of patch centered at pixel value
def get_patch(pixel):
    startx = int(max(0, pixel[0] - patchwidth/2))
    starty = int(max(0, pixel[1] - patchwidth/2))
    return (startx, starty)

# Based on starting pixels of two patches, slices the image and returns the difference of the two
def patch_difference(I, patch1, patch2):
    patch1_img = I[patch1[0]:patch1[0] + patchwidth, patch1[1]:patch1[1] + patchwidth]
    # Account for when patch reached edge of image and must be truncated
    adjustedwidth = patch1_img.shape[0]
    adjustedheight = patch1_img.shape[1]
    patch2_img = I[patch2[0]:patch2[0] + adjustedwidth, patch2[1]:patch2[1] + adjustedheight]
    return patch1_img - patch2_img


### All of the following functions are attempts to recreate the functions defined in the paper

##########################################################################
############################ psi/alpha ###################################

# psi = sparsity prior
# This effectively removes rain streaks, but background details as well
# Since we could not get psi fully working we just use our own basic smoothing method
def regularize_psi(B):
    return alpha(B)

# Go to each patch in the picture and average values to smooth picture
def alpha(B):
    return cv2.medianBlur(B, 3)

##########################################################################
############################### phi ######################################

# phi = Rain direction prior
# Helps put back in background details that could be mistaken for streaks

epsilon_1 = 0.0001 # Constant to avoid division by 0

def regularize_phi(B):
    rows = B.shape[0]
    cols = B.shape[1]
    # loop through each pixel and assign value to pixel in output
    output = np.zeros((rows, cols), float)
    for i in range(rows-1):
        for j in range(cols-1):
            term = theta_i(B, i, j) + epsilon_1
            output[i][j] = (1 / term)
    return output

# compares rain direction and pixel gradient
def theta_i(B, x, y):
    gradient = pixel_gradient(B, x, y)
    numerator = np.dot(global_rain_vector(), gradient)
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
    R_norm = img_normalize(R)
    dx, dy = partial_both_grayscale(R_norm)
    # loop through each pixel and add to sum
    output = np.zeros((rows, cols), float)
    for i in range(rows-2):
        for j in range(cols-2):
            pixel = (i, j)
            dx_pixel = dx[pixel]
            dy_pixel = dy[pixel]
            biggamma = gamma_i(I, pixel)
            term1 = weight_x(I, biggamma, dx_pixel) * (dx_pixel ** 2)
            term2 = weight_y(I, biggamma, dy_pixel) * (dy_pixel ** 2)
            output[i][j] = (term1 + term2)
    return output

# Smoothing weight on pixel i in x direction
def weight_x(I, biggamma, partial):
    term = partial * biggamma
    return abs(term) ** eta

# Smoothing weight on pixel i in y direction
def weight_y(I, biggamma, partial):
    term = partial * biggamma
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

