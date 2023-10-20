#from scipy.spatial.distance import cdist
from pickle import NONE
import numpy as np


class StructuralElement:
    
    def __init__(self, global_nodes, area, elastic_modulus):
        self.global_nodes = global_nodes
        self.area = area
        self.elastic_modulus = elastic_modulus
        self.element_number = 0
        
    def get_global_nodes(self):
        return self.global_nodes
        
    def get_cross_section_area(self):
        return self.area

    def get_elastic_modulus(self):
        return self.elastic_modulus

    def set_element_number(self, element_number):
        self.element_number = element_number
        return

    def get_element_number(self):
        return self.element_number
                                                                                                                            

class TrussElement(StructuralElement):
    
    def __init__(self):
        self.global_nodes = None
        self.area = None
        self.elastic_modulus = None
        self.element_number = 0
    
    def __init__(self, global_nodes=None, area=None, elastic_modulus=None):
        if global_nodes is None:
            self.global_nodes=None
        else:
            self.global_nodes = global_nodes

        if area is None:
            self.area=None
        else:
            self.area = area

        if elastic_modulus is None:
            self.elastic_modulus=None
        else:
            self.elastic_modulus = elastic_modulus

        self.element_number = -1

    # The element stiffness matrix assumes a 2D structural system with 
    # the DOF UX, UY, and RZ
    def get_element_stiffness_matrix(self):
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        element_length = np.linalg.norm(element_vector)
        print(element_vector)
        # Cosine of angle relative to X-axis
        c = np.dot(element_vector, np.array([1, 0])) / element_length
        print(c)
        # Sine of angle relative to X-axis 
        s = np.cross(np.array([1, 0]), element_vector) / element_length
        print(s)
        #                   NODE 1     |     NODE 2 
        #                 UX     UY RZ |   UX     UY RZ
        k = np.array([[ c**2,   c*s, 0, -c**2,  -c*s, 0],
                      [  c*s,  s**2, 0,  -c*s, -s**2, 0],
                      [    0,     0, 0,     0,     0, 0],
                      [-c**2,  -c*s, 0,  c**2,   c*s, 0],
                      [ -c*s, -s**2, 0,   c*s,  s**2, 0],
                      [    0,     0, 0,     0,     0, 0]
                     ])
        
        return k*self.area*self.elastic_modulus/element_length
                
