import numpy as np
import cv2
import os
import functions

lambda_1 = 1
lambda_2 = 1
lambda_3 = 1

I = cv2.imread('1.png')

f = functions
f.calc_global_rain_direction(I)

B = I
R = B

## Goal is to minimize this term
term0 = f.frobenius_norm(I - B - R) ** 2 # Gets an output
term1 = lambda_1 * f.regularize_psi(B)
term2 = lambda_2 * f.regularize_phi(B) # Gets an output
term3 = lambda_3 * f.regularize_omega(R)
term = term0 + term1 + term2 + term3
