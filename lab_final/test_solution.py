#!/usr/bin/env python3

'''
This code contains the theoretical ground truth for the MMS Validation.

'''

import numpy as np

# Theoretical solution for the method of manufactured solution test case at t = 0.1
# Parameters: 
#   Target function: U = 100 * t * sin(pi * z)
#   Time: t = 0.1
#   Grid: z = 0.0 to 1.0 with dz = 0.1 (11 points)

mms_ground_truth = [
    0.000000,   # z = 0.0
    3.090170,   # z = 0.1
    5.877853,   # z = 0.2
    8.090170,   # z = 0.3
    9.510565,   # z = 0.4
    10.000000,  # z = 0.5
    9.510565,   # z = 0.6
    8.090170,   # z = 0.7
    5.877853,   # z = 0.8
    3.090170,   # z = 0.9
    0.000000    # z = 1.0
]

# Convert to a numpy array for easy comparison
mms_ground_truth = np.array(mms_ground_truth)