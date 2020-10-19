import numpy as np
import cv2
import math

##########################################################################
############################## common ####################################

global_rain_direction = [0, 0]

def frobenius_norm(I):
    return np.linalg.norm(I, ord=None, axis=None, keepdims=False)

def normalize(v):
    return v / np.sqrt(np.sum(v ** 2))

# TODO: This one
def calc_global_rain_direction(I):
    global global_rain_direction
    global_rain_direction = [1, 1]

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
            gx[i-1, j-1] = (img[i, j+1]-img[i, j-1])/2.0
            gy[i-1, j-1] = (img[i+1, j]-img[i-1, j])/2.0
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


def pixel_gradient(I, x, y):
    vector = []
    partx = (I[x, y + 1] - I[x, y - 1]) / 2.0
    party = (I[x + 1, y] - I[x - 1, y]) / 2.0
    vector.append((partx[0] + partx[1] + partx[2]) / 3)
    vector.append((party[0] + party[1] + party[2]) / 3)
    return [vector[0], vector[1]]

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
# helps put back in background details that could be mistaken for streaks

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
# push scene details from R back into B

eta = 1.2 # sensitivity parameter

def regularize_omega(R):
    rows = R.shape[0]
    cols = R.shape[1]
    # loop through each pixel and add to sum
    sum = 0
    for i in range(rows - 1):
        for j in range(cols - 1):
            pixel = R[i, j]
            term1 = weight_x(R, pixel) * (partial_x(R) * pixel) ** 2
            term2 = weight_y(R, pixel) * (partial_y(R) * pixel) ** 2
            sum += (term1 + term2)
    return sum

# Smoothing weight on pixel i in x direction
def weight_x(R, pixel):
    term = partial_x(R) * gamma_i(R, pixel)
    return abs(term) ** eta

# Smoothing weight on pixel i in y direction
def weight_y(R, pixel):
    term = partial_y(R) * gamma_i(R, pixel)
    return abs(term) ** eta

# TODO: This one
# Similarity map
def gamma_i(R, pixel):
    vector = []
    # go through each extracted rain patch and do something, return min
    return min(vector)

