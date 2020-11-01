import numpy as np
import cv2
import os
import functions
import math
import random

# Declaring parameter values, most of them are constant

lambda_1 = 1.0
lambda_2 = 0.0005 # Tuned parameter in range [0.0001, 0.001]
lambda_3 = 0.01

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

gradient_angles = f.grad_angle(gx, gy) # Currently the result is in range 0 to pi/2. Paper states should be in range 0 to pi

#print(gradient_angles.max())

# cv2.imshow('Gradient Angles', gradient_angles)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Construct array of windows which each have some value delta

n_row = gradient_angles.shape[0]
n_col = gradient_angles.shape[1]

# Windows is a list of 2 tuples, the coords of the top-left value in each window
windows = list()
# Deltas is a parallel list of delta values
deltas = list()

# This loop implements a sliding window size W_r x W_r and with stride W_s
# For each window a histogram with 10 bins of gradient angles is calculated
# Delta for each window will equal the bin with the most elements and its two adjacent bins
# After delta is calculated for each window, the N windows with the highest delta are selected as rainy window
y = 0
x = 0
while True:
    # Construct a histogram of angle values for each window
    histogram = np.zeros(10, int)
    # Window size is W_r x W_r
    # x and y indicate the top-left corner of the window
    for ix in range(x, x+W_r):
        for iy in range(y, y+W_r):
            angle = gradient_angles[ix, iy]
            for anglebin in range(1, 11):
                # This looks complicated but just checks which range the angle resides in
                # For example, when bin = 1, it checks if angle is in range [0, pi/10)
                # When bin = 2, it checks if angle is in range [pi/10, 2*pi/10), etc.
                if (anglebin - 1) * math.pi / 10 <= angle < anglebin * math.pi / 10:
                    histogram[anglebin-1] += 1
                    break # Break after finding correct bin
    # Now must calculate delta, which equals the number of values in the bin with the most values, and its two adjacent bins
    maxindex = np.argmax(histogram)
    delta = histogram[maxindex]
    if maxindex > 0: # Consider previous adjacent bin unless this is the first bin
        delta += histogram[maxindex - 1]
    if maxindex < 9: # Consider next adjacent bin unless this is the last bin
        delta += histogram[maxindex + 1]
    window = (x, y)
    windows.append(window) # Add to list of windows
    deltas.append(delta) #Add to list of deltas
    # Now must determine how to increment x and y values
    if x + W_s + W_r > n_row - 1 and y + W_s + W_r > n_col - 1: # If next window is outside image in both dimensions, end has been reached
        break
    elif x + W_s + W_r > n_row - 1: # If next window would be outside image in x direction, reset x direction and go to next row
        x = 0
        y += W_s
    else: # Otherwise just increment x direction
        x += W_s

# Now we want to find the index of the max deltas, and get the windows with the same index
# These will be considered the rainy windows
rainywindows = list()
for i in range(0, N):
    # Get the max N delta values, append them to rainy windows, and remove those values from both lists
    maxindex = np.argmax(deltas)
    rainywindows.append(windows.pop(maxindex))
    deltas.pop(maxindex)

# Test to see canny edge detector results
# canny = cv2.Canny(I, 0, 100)
# canny_gray = cv2.Canny(I_gray, 0, 100)
# cv2.imshow('Image Edges', canny)
# cv2.imshow('Gray Image Edges', canny_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Now the goal is to find the longest rain streak (edge) in each window and take their slope
# The median of the list of slopes is the global rain direction
I_windows = I.copy()
rainslopes = list()
for window in rainywindows:
    # Perform canny edge detection on each window to hopefully get edges for rain streaks
    cropped = I[window[1]:window[1]+W_r, window[0]:window[0]+W_r]
    cropped_canny = cv2.Canny(cropped, 0, 100) # Experimentally, this threshold seems to be the best at showing the rain streaks as edges

    # cv2.imshow('Rainy Window', cropped)
    # cv2.imshow('Rainy Window Edges', cropped_canny)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Now use hough transform to identify lines (rain streaks) and try to find longest one
    lines = cv2.HoughLines(cropped_canny, 1, np.pi / 180, 20)
    if not (lines is None): # Once window identification works, this should never be false
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
        rainslopes.append(maxline[0][1]) # Add the theta of the maximum length line to the list of rain slopes

    # Draw rectangle over each window to identify them
    bottomright = (window[0] + W_r, window[1] + W_r)
    cv2.rectangle(I_windows, window, bottomright, (0, 0, 255), 2)

# Take median rain slope from the list of rain slopes of the longest line in each rainy window, this is the global rain direction
global_rain_direction = np.median(rainslopes)
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
cv2.imshow('Rainy Windows and Patches', I_windows)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit(0)

# Initialize vars for use in minimization code
B = I.copy()
R = I - B
f.global_rain_direction = global_rain_direction
f.rainy_patches = rainy_patches
f.patchwidth = patchwidth


## Goal is to minimize this term
term0 = f.frobenius_norm(I - B - R) ** 2
term1 = lambda_1 * f.regularize_psi(B) # Need to add code for calculation of sparse code in each patch
term2 = lambda_2 * f.regularize_phi(B)
term3 = lambda_3 * f.regularize_omega(R)
term = term0 + term1 + term2 + term3
