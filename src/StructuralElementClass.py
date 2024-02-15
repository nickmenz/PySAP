#from scipy.spatial.distance import cdist
from pickle import NONE
import numpy as np
import Utility as util


class StructuralElement:
    
    def __init__(self, global_nodes, area, elastic_modulus):
        self.global_nodes = global_nodes
        self.area = area
        self.elastic_modulus = elastic_modulus
        self.element_number = -1
        self.local_dof_deformation = None
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        element_length = np.linalg.norm(element_vector)
        self.element_length = element_length
        # positive = CCW from global X-axis
        self.angle_relative_to_global_x = np.arccos(np.dot(element_vector, np.array([1, 0])) / element_length)

        
    def get_global_nodes(self):
        return self.global_nodes
        
    def get_cross_section_area(self):
        return self.area

    def get_elastic_modulus(self):
        return self.elastic_modulus

    def get_angle_relative_to_global_x(self):
        return self.angle_relative_to_global_x

    def get_element_length(self):
        return self.element_length

    def set_element_number(self, element_number):
        self.element_number = element_number
        return

    def get_element_number(self):
        return self.element_number

    def get_dof_deformation(self):
        # [u1, v1, R1, u2, v2, R2]
        return self.local_dof_deformation
    
    def process_element_results(self):
        node1_deformation = self.global_nodes[0].get_dof_deformation()
        node2_deformation = self.global_nodes[1].get_dof_deformation()
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        element_length = np.linalg.norm(element_vector)
        elem_angle = np.arccos(np.dot(element_vector, np.array([1, 0])) / element_length)
        node1_dof_deformation = np.dot(util.get_nodal_dof_rotation_matrix(elem_angle).T, node1_deformation)
        node2_dof_deformation = np.dot(util.get_nodal_dof_rotation_matrix(elem_angle).T, node2_deformation)
        self.local_dof_deformation = np.concatenate((node1_dof_deformation, node2_dof_deformation))
                                                                                                                            

class TrussElement(StructuralElement):
    
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
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        element_length = np.linalg.norm(element_vector)
        self.element_length = element_length
        # positive = CCW from global X-axis
        self.angle_relative_to_global_x = np.arccos(np.dot(element_vector, np.array([1, 0])) / element_length)

    # The element stiffness matrix assumes a 2D structural system with 
    # the DOF UX, UY, and RZ
    def get_element_stiffness_matrix(self):
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        element_length = np.linalg.norm(element_vector)
        # Cosine of angle relative to X-axis
        c = np.dot(element_vector, np.array([1, 0])) / element_length
        # Sine of angle relative to X-axis 
        s = np.cross(np.array([1, 0]), element_vector) / element_length
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

    def get_axial_force(self, discretization):
        if self.local_dof_deformation == None:
            print("Structure has not yet been solved!")
            return 0
        else:
            return np.full(discretization, (self.local_dof_deformation[3] - self.local_dof_deformation[0])*self.elastic_modulus*self.area/self.element_length)

    def get_shape_functions(self, discretization):
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        L = np.linalg.norm(element_vector)
        x = np.linspace(0, L, discretization)
        N1 = 1 - x
        N2 = 1 - x
        N3 = 0
        N4 = x
        N5 = x
        N6 = 0
        return np.array([N1, N2, N3, N4, N5, N6])


                
class BeamElement(StructuralElement):
    def __init__(self, global_nodes=None, area=None, elastic_modulus=None, moment_of_inertia=None):
        self.global_nodes = global_nodes
        self.area = area
        self.elastic_modulus = elastic_modulus
        self.moment_of_inertia = moment_of_inertia
        self.element_number = -1
        self.node1_dof_deformation = None
        self.node2_dof_deformation = None
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        element_length = np.linalg.norm(element_vector)
        self.element_length = element_length
        # positive = CCW from global X-axis
        self.angle_relative_to_global_x = np.arccos(np.dot(element_vector, np.array([1, 0])) / element_length)
        self.distributed_load_magnitude = 0
    

    def get_element_stiffness_matrix(self):
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        L = np.linalg.norm(element_vector)
        # Cosine of angle relative to X-axis
        c = np.dot(element_vector, np.array([1, 0])) / L
        # Sine of angle relative to X-axis 
        s = np.cross(np.array([1, 0]), element_vector) / L
        t1 = self.area*L**2 / self.moment_of_inertia
        s2 = s**2
        c2 = c**2

        k11 = t1*c2 + 12*s2
        k12 = (t1-12)*c*s
        k22 = t1*s2 + 12*c2
        k13 = -6*L*s
        k23 = 6*L*c
        k33 = 4*L**2
        k14 = -(t1*c2 + 12*s2)
        k24 = -(t1-12)*c*s
        k34 = 6*L*s
        k44 = t1*c2 + 12*s2
        k15 = -(t1-12)*c*s
        k25 = -(t1*s2 + 12*c2)
        k35 = -6*L*c
        k45 = (t1-12)*c*s
        k55 = (t1*s2 + 12*c2)
        k16 = -6*L*s
        k26 = 6*L*c
        k36 = 2*L**2
        k46 = 6*L*s
        k56 = -6*L*c
        k66 = 4*L**2

        #                   NODE 1    |   NODE 2 
        #                UX,  UY,  RZ,| UX,  UY,  RZ,
        k = np.array([[ k11, k12, k13, k14, k15, k16],
                      [ k12, k22, k23, k24, k25, k26],
                      [ k13, k23, k33, k34, k35, k36],
                      [ k14, k24, k34, k44, k45, k46],
                      [ k15, k25, k35, k45, k55, k56],
                      [ k16, k26, k36, k46, k56, k66],
                     ])

        return k*self.moment_of_inertia*self.elastic_modulus/L**3
    
    def get_axial_force(self, discretization):
        if not self.local_dof_deformation.any():
            print("Structure has not yet been solved!")
            return 0
        else:
            return np.full(discretization, (self.local_dof_deformation[3] - self.local_dof_deformation[0])*self.elastic_modulus*self.area/self.element_length)

    def get_bending_moment(self, discretization):
        if not self.local_dof_deformation.any():
            print("Structure has not yet been solved!")
            return 0
        else:
            #shape_functions = self.get_shape_functions(discretization)
            #lateral_disp = self.local_dof_deformation[1] * shape_functions[1] + self.local_dof_deformation[4] * shape_functions[4]
            #spacing = self.element_length/discretization
            #return np.gradient(np.gradient(lateral_disp, spacing, edge_order=2), spacing, edge_order=2)
            shape_functions = self.get_shape_functions_2nd_derivative(discretization)
            return self.elastic_modulus*self.moment_of_inertia*(self.local_dof_deformation[1] * shape_functions[1] + self.local_dof_deformation[4] * shape_functions[4])

    def get_shear(self, discretization):
        if not self.local_dof_deformation.any():
            print("Structure has not yet been solved!")
            return 0
        else:
            shape_functions = self.get_shape_functions_3rd_derivative(discretization)
            return self.elastic_modulus*self.moment_of_inertia*(self.local_dof_deformation[1] * shape_functions[1] + self.local_dof_deformation[4] * shape_functions[4])

    def get_lateral_displacement(self, discretization):
        if not self.local_dof_deformation.any():
            print("Structure has not yet been solved!")
            return 0
        else:
            shape_functions = self.get_shape_functions(discretization)
            return self.local_dof_deformation[1] * shape_functions[1] + self.local_dof_deformation[4] * shape_functions[4] 
    
    def get_shape_functions(self, discretization):
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        L = np.linalg.norm(element_vector)
        x = np.linspace(0, L, discretization)
        N1 = (1 - x) / L
        N2 = 1 - 3*x**2/L**2 + 2*x**3/L**3
        N3 = x - 2*x**2/L + x**3/L**2
        N4 = x / L
        N5 = 3*x**2/L**2 - 2*x**3/L**3
        N6 = -x**2/L + x**3/L**2
        return np.array([N1, N2, N3, N4, N5, N6])
    
    def get_shape_functions_2nd_derivative(self, discretization):
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        L = np.linalg.norm(element_vector)
        x = np.linspace(0, L, discretization)
        N1 = x*0
        N2 = -6/L**2 + 12*x/L**3
        N3 = -4*x/L + 6*x/L**2
        N4 = x*0
        N5 = 6/L**2 - 12*x/L**3
        N6 = -2/L + 6*x/L**2
        return np.array([N1, N2, N3, N4, N5, N6])

    def get_shape_functions_3rd_derivative(self, discretization):
        element_vector = np.subtract(self.global_nodes[1].get_coordinates(), self.global_nodes[0].get_coordinates())
        L = np.linalg.norm(element_vector)
        xd = np.full(discretization, 1) # dummy array so that each shape function will be an array of shape [disc]
        N1 = xd*0
        N2 = xd*(12/L**3)
        N3 = xd*(-4/L + 6/L**2)
        N4 = xd*0
        N5 = xd*(-12/L**3)
        N6 = xd*(6/L**2)
        return np.array([N1, N2, N3, N4, N5, N6])

    def apply_distributed_load(self, distributed_load_magnitude):
        """_summary_

        Args:
            distributed_load_magnitude (_type_): _description_
        """        
        self.distributed_load_magnitude = distributed_load_magnitude
        
    def get_distributed_load_magnitude(self):
        return self.distributed_load_magnitude

    def get_equivalent_nodal_load_vector(self, node):
        q = self.distributed_load_magnitude
        L = self.element_length
        if node == self.global_nodes[0]:
            return np.array([0, -q*L/2, -q*L**2/12])
        else:
            return np.array([0, -q*L/2, q*L**2/12])