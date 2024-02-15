from typing import int

import numpy as np

class Node:
    def __init__(self, coordinates, dof_boundary_conditions):
        self.coordinates = np.array(coordinates,dtype=np.float32)
        self.deformed_coordinates = np.array(coordinates)
        self.dof_deformation = np.array([0.0, 0.0, 0.0])
        # 0 if active (unconstrained) DOF, 1 if inactive (constrained)
        self.dof_boundary_conditions = np.array(dof_boundary_conditions, dtype=np.int8)   
        self.node_number = -1
        self.load_vector = np.array([0., 0., 0.])

     
    def get_coordinates(self):
        return self.coordinates

    def get_deformed_coordinates(self):
        return self.deformed_coordinates

    def set_dof_boundary_conditions(self, dof_boundary_conditions):
        self.dof_boundary_conditions = np.array(dof_boundary_conditions, dtype=np.int8) 

    def get_dof_boundary_conditions(self):
        return self.dof_boundary_conditions

    def set_node_number(self, node_number):
        self.node_number = node_number

    def save_dof_deformation(self, deformed_dof_displacements):
        self.dof_deformation = deformed_dof_displacements
        self.deformed_coordinates[0] = self.coordinates[0] + deformed_dof_displacements[0]
        self.deformed_coordinates[1] = self.coordinates[1] + deformed_dof_displacements[1]
        
    def get_dof_deformation(self):
        return self.dof_deformation

    def get_node_number(self):
        return self.node_number

    def apply_point_load(self, load_vector):
        self.load_vector = load_vector


    def get_nodal_point_load_vector(self):
        return self.load_vector




