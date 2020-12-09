import numpy as np
import cv2
import os
import functions
import math
import random

# Declaring constant parameter values

lambda_1 = 1.0
lambda_2 = 0.1
lambda_3 = 0.1

W_r = 31 # Window size
W_s = 8 # Window step size
N = 20 # Number of rainy windows
patchwidth = 7 # Width of rainy patches

T = 2
K = 25 # Tuned parameter in range [4, 50]

# Read in image, compute gradients, gradient angle, attempt to extract rainy patches from gradient angle

I = cv2.imread('TestUmbrella.png')

I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Original image', I)
# cv2.imshow('Gray image', I_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

f = functions

I_gray_norm = f.img_normalize(I_gray)
gx, gy = f.partial_both_grayscale(I_gray)

# cv2.imshow('x derivative', gx)
# cv2.imshow('y derivative', gy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gradient_angles = f.grad_angle(gx, gy)

# print(gradient_angles.max())

# cv2.imshow('Gradient Angles', gradient_angles)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Test to see canny edge detector results
# canny = cv2.Canny(I, 0, 100)
canny_gray = cv2.Canny(I_gray, 0, 80)
# cv2.imshow('Image Edges', canny)
# cv2.imshow('Gray Image Edges', canny_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Construct array of windows which each have some value delta

n_row = gradient_angles.shape[0]
n_col = gradient_angles.shape[1]

# Windows is a list of 2 tuples, the coords of the top-left value in each window
windows = list()
# Deltas is a parallel list of delta values
deltas = list()


# Use our own method where we find hough lines in a window with at least 7 lines
# Check each of their angles and save delta as the standard deviation of the angles
# This will make it so that windows with several lines going in the same direction will be considered
y = 0
x = 0
while True:
    # Window size is W_r x W_r
    # x and y indicate the top-left corner of the window
    cropped = canny_gray[y:y+W_r, x:x+W_r]
    # Now use hough transform to identify lines (rain streaks) and try to find longest one
    lines = cv2.HoughLines(cropped, 1, np.pi / 180, 20)
    delta = 1000000.0
    if not(lines is None) and lines.shape[0] > 8: # Want at least 7 lines to consider this window
        angles = list()
        for line in lines:
            angles.append(line[0][1])
        delta = np.std(angles) # Find the standard deviation of the angles, low std dev means lines are going in the same direction
    window = (x, y)
    windows.append(window) # Add to list of windows
    deltas.append(delta) # Add to list of deltas
    # Now must determine how to increment x and y values
    if x + W_s + W_r > n_row - 1 and y + W_s + W_r > n_col - 1: # If next window is outside image in both dimensions, end has been reached
        break
    elif x + W_s + W_r > n_row - 1: # If next window would be outside image in x direction, reset x direction and go to next row
        x = 0
        y += W_s
    else: # Otherwise just increment x direction
        x += W_s

# Now we want to find the index of the minimum deltas, and get the windows with the same index
# These will be considered the rainy windows
rainywindows = list()
for i in range(0, N):
    # Get the min N delta values, append them to rainy windows, and remove those values from both lists
    minindex = np.argmin(deltas)
    rainywindows.append(windows.pop(minindex))
    deltas.pop(minindex)

# Now the goal is to find the longest rain streak (edge) in each window and take their slope
# The median of the list of slopes is the global rain direction
I_windows = I.copy()
rainslopes = list()
majority = 0
for window in rainywindows:
    # Perform canny edge detection on each window to hopefully get edges for rain streaks
    cropped = canny_gray[window[1]:window[1]+W_r, window[0]:window[0]+W_r]
    # cv2.imshow('Rainy Window', cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Now use hough transform to identify lines (rain streaks) and try to find longest one
    lines = cv2.HoughLines(cropped, 1, np.pi / 180, 20)
    if not (lines is None): # This should never be false
        lengths = list()
        for line in lines:
            # Do some calculations to find two points (start and end) of the line and calculate its length
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000 * a)) # Line start
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000 * a)) # Line end
            length = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt2[0])**2) # Calculates length of line
            lengths.append(length) # Add to length list
        maxindex = np.argmax(lengths) # Get maximum length
        maxline = lines[maxindex] # This is the line with maximum length
        thetamax = maxline[0][1]
        rhomax = maxline[0][0]
        majorityadd = 1
        if rhomax > 0: # If rho is positive, consider slope as negative
            majorityadd = -1
        rainslopes.append(thetamax) # Add the theta of the maximum length line to the list of rain slopes
        majority += majorityadd

    # Draw rectangle over each window to identify them
    bottomright = (window[0] + W_r, window[1] + W_r)
    cv2.rectangle(I_windows, window, bottomright, (0, 0, 255), 2)


# If some slopes are opposite direction from majority, are likely an error, want to ignore
multiplier = 1
if majority < 0:
    multiplier = -1

# Take median rain slope from the list of rain slopes of the longest line in each rainy window, this is the global rain direction
global_rain_direction = np.max(rainslopes) * multiplier
# print(global_rain_direction)
# Draw line representing global rain direction on image
if global_rain_direction > 0: # positive slope, draw from bottom left
    bottomleft = (0, I_windows.shape[0])
    adjust_amount = math.ceil(I_windows.shape[0] / global_rain_direction) # Want to draw line slightly past edge
    adjusted_y = int(adjust_amount*global_rain_direction) # Calculate y value of point based on slope and how far to go
    adjusted = (bottomleft[0] + adjust_amount, bottomleft[1] - adjusted_y) # Adjust x forward and y up to make a sloped line
    cv2.line(I_windows, bottomleft, adjusted, (0, 255, 0), 1)
else: # negative slope, draw from bottom right
    abs_slope = abs(global_rain_direction)
    bottomright = (I_windows.shape[1], I_windows.shape[0])
    adjust_amount = max(math.ceil(I_windows.shape[0] / abs_slope), math.ceil(I_windows.shape[1] / abs_slope)) # Want to draw line slightly past edge
    adjusted_y = int(adjust_amount*abs_slope) # Calculate y value of point based on slope and how far to go
    adjusted = (bottomright[0] - adjust_amount, bottomright[1] - adjusted_y) # Adjust x backward and y up to make a sloped line
    cv2.line(I_windows, bottomright, adjusted, (0, 255, 0), 1)


# Next step: Extract rainy patches of size patchwidth x patchwidth
# Want total of 20N rainy patches, chosen randomly
rainy_patches = list()
for i in range(0, 20*N):
    windowindex = random.randint(0, N-1) # Random int in range [0, N-1] to select random window
    selectedwindow = rainywindows[windowindex]
    # Random x, y coordinates, make sure they don't go past edge
    startx = random.randint(0, W_r - patchwidth - 1)
    starty = random.randint(0, W_r - patchwidth - 1)
    patch = (startx, starty)
    rainy_patches.append(patch)

    # Draw rectangle over each patch to identify them
    topleft_patch = (selectedwindow[0] + startx, selectedwindow[1] + starty)
    bottomright_patch = (selectedwindow[0] + startx + patchwidth, selectedwindow[1] + starty + patchwidth)
    #cv2.rectangle(I_windows, topleft_patch, bottomright_patch, (0, 0, 255), 2)

# Show image with windows, lines, and patches drawn over it
# cv2.imshow('Rainy Windows and Patches', I_windows)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Initialize vars for use in minimization code
B = I_gray_norm.copy()
B = B.astype('float32') # Must convert everything to float32 for opencv functions to work
R = I_gray_norm - B
f.global_rain_direction = global_rain_direction
f.rainy_patches = rainy_patches
f.patchwidth = patchwidth
alpha = f.img_normalize(f.alpha(B))
beta = .01
H = B.copy() # Lagrange multiplier of linear constraint. Should equal (B - D * alpha) but D will initially equal 0 so initial just = B

# Include to see visual result for psi
# cv2.imshow('Psi/alpha output', alpha)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Include to see visual result for phi
# phi = f.regularize_phi(B)
# cv2.imshow('Phi output', phi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Include to see visual result for omega
# omega = f.regularize_omega(B, I)
# cv2.imshow('Omega output', omega)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Goal is to minimize this term
# term0 = f.frobenius_norm(I - B - R) ** 2
# term1 = lambda_1 * f.regularize_psi(B) # Need to add code for calculation of sparse code in each patch
# term2 = lambda_2 * f.regularize_phi(B)
# term3 = lambda_3 * f.regularize_omega(R, I)
# term = term0 + term1 + term2 + term3

# Iterative code for optimization
for t in range(0, 10):
    # Update B step 1: update B_t+1
    firstpart = f.img_normalize(I_gray_norm - B - R)
    secondpart = lambda_2 * f.img_normalize(f.regularize_phi(B))
    D = lambda_3 * f.img_normalize(f.regularize_omega(R, I_gray_norm))
    tonormalize_1 = B - D * alpha - (1 / beta) * H
    thirdpart = f.img_normalize(tonormalize_1)
    B = B + firstpart + secondpart + (beta / 2) * thirdpart
    B = f.img_normalize(B)
    B = B.astype('float32')
    # Update B step 2: update alpha_t+1
    tonormalize_2 = B - D * alpha - (1 / beta) * H
    alpha = f.img_normalize(tonormalize_2) + lambda_1 * f.img_normalize(f.regularize_psi(B))
    alpha = f.img_normalize(alpha)
    alpha = alpha.astype('float32')
    # Update B step 3: Update H_t+1
    H = H + beta * (B - D * alpha)
    H = f.img_normalize(H)
    H = H.astype('float32')
    # Update R
    R = I_gray_norm - B
    R = f.img_normalize(R)
    R = R.astype('float32')

    # term1 = lambda_1 * f.img_normalize(f.regularize_psi(B))
    # term2 = lambda_2 * f.img_normalize(f.regularize_phi(B))
    # term3 = lambda_3 * f.img_normalize(f.regularize_omega(R, I_gray_norm))
    # B = term1 + term2 + term3
    # B = f.img_normalize(B)
    # B = B.astype('float32')
    # R = I_gray_norm - B
    # R = f.img_normalize(R)
    # R = R.astype('float32')


# Show Original
cv2.imshow('Original', I_gray_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show Result
cv2.imshow('B Result', B)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show Result
cv2.imshow('R Result', R)
cv2.waitKey(0)
cv2.destroyAllWindows()
