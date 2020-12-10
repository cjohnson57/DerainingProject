# Deraining Project
Project to remove rain streaks from images for EE 7700

The project is based off of the paper "Joint Bi-layer Optimization for Single-image Rain Streak Removal"

https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.pdf

It utilizes Python with OpenCV and Numpy

[main.py](main.py): Code to extract rainy windows and patches from image, calculate global rain direction, and run the optimization step to produce a derained result.

[functions.py](functions.py): Contain some sub functions for use in main, mainly for calculation of image derivatives, gradient directions/angles, and the priors defined in the author's paper.

[Project Report.pdf](Project%20Report.pdf): Contains details of the author's design, my implementation of their design, changes I made to their design which worked better for me, and images of some results.
