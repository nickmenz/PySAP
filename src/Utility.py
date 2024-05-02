import numpy as np


# By default, gives rotation matrix that can be multiplied by the DOF
# vector to find the DOF of a node in 2D rotated CCW by theta in the
# same coordinate system. Note that the transpose of the matrix can be
# multiplied by the DOF vector to yield the DOF in a second coordinate system 
# that is oriented theta (CCW) from the original
def get_nodal_dof_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [            0,              0, 1]
                     ])

# Gives rotation matrix that can be multiplied by 2D coordinates or vectors
# to find the coordinates/vector rotated CCW by theta in the 
# same coordinate system. Note that the transpose of the matrix can be 
# multiplied by the coordinates/vector to yield coordinates/vector in a 
# second coordinate system that is oriented theta (CCW) from the original
def get_2D_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]
                     ])

