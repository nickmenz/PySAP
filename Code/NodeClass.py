import numpy as np

class Node:
    def __init__(self, coordinates, dof_boundary_conditions):
        self.coordinates = np.array(coordinates)
        # 0 if active (unconstrained) DOF, 1 if inactive (constrained)
        self.dof_boundary_conditions = np.array(dof_boundary_conditions, dtype=np.int8)   
        self.node_number = -1


    def get_coordinates(self):
        return self.coordinates

    def set_dof_boundary_conditions(self, dof_boundary_conditions):
        self.dof_boundary_conditions = np.array(dof_boundary_conditions, dtype=np.int8) 

    def get_dof_boundary_conditions(self):
        return self.dof_boundary_conditions

    def set_node_number(self, node_number):
        self.node_number = node_number
        return

    def get_node_number(self):
        return self.node_number




