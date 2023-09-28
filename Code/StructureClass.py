import StructuralElementClass as el
import NodeClass as nd
import numpy as np


class Structure:
    
    # This implementation assumes a 2D structural system; i.e., the only DOF
    # are UX, UY, and RZ
    def __init__(self, name):
        self.name = name
        self.number_of_nodes = 0
        self.node_list = []
        self.number_of_elements = 0
        self.element_list = []
        self.number_of_point_loads = 0
        self.point_load_list = []
        self.number_of_distributed_loads = 0
        self.distributed_load_list = []
        self.NUM_DOF_PER_NODE = 3

    
    def print_name(self):
        print(self.name)
        return
        
    def add_node(self, node):
        self.node_list.append(node)
        self.number_of_nodes += 1
        return

    def list_nodes(self):
        for node, i  in enumerate(self.node_list):
            print("node " + str(i) + " = " + str(node.get_coords()))
        return

    def add_element(self, element):
        self.element_list.append(element)
        self.number_of_elements += 1
        return

    def list_elements(self):
        for element in self.element_list:
            print(element)
        return

    def add_point_load(self, load):
        self.point_load_list.append(load)
        self.number_of_point_loads += 1
        return

    def list_point_loads(self):
        for point_load in self.point_load_list:
            print(point_load)
        return

    def add_distributed_load(self, load):
        self.distributed_load_list.append(load)
        self.number_of_distributed_loads += 1
        return
    
    def list_distributed_loads(self):
        for distributed_load in self.distributed_load_list:
            print(distributed_load)
        return

    def solve(self):
       # Number all elements and nodes
        for i, node in enumerate(self.node_list):
            node.set_node_number(i)

        for i, element in enumerate(self.element_list):
            element.set_element_number(i)

        # Define the ID array where 1 = constrained DOF, 0 = unconstrained DOF
        identification_array = self.create_identification_array()
        print("ID array = ")
        print(identification_array)
        # Convert ID array so that 0 = constrained DOF, other numbers indicate numbering of DOF
        identification_array_converted, num_unconstrained_dof = \
            self.convert_identification_array_to_dof_numbering_array(identification_array)
        print("ID array converted = ")
        print(identification_array_converted)
        
        global_K = np.zeros((num_unconstrained_dof, num_unconstrained_dof))
        # Assemble the global stiffness matrix from the individual element stiffness matrices
        for element in self.element_list:
            element_connectivity_array = self.get_connectivity_array(element, identification_array_converted)
            print(f"Element connectivity array for element {element.get_element_number()} = ")
            print(element_connectivity_array)
            element_k = element.get_element_stiffness_matrix()
            print(f"Element k for element {element.get_element_number()} = ")
            print(element_k)
            for row in range(element_k.shape[0]):
                for col in range(element_k.shape[1]):
                    if((element_connectivity_array[row] != 0) and (element_connectivity_array[col] != 0)):              
                        global_K[element_connectivity_array[row] - 1, element_connectivity_array[col] - 1] += element_k[row, col]
        print("Assembled global K = ")
        print(global_K)  
        
        # Assemble the global force vector
        global_F = np.zeros((num_unconstrained_dof))
        for point_load in self.point_load_list:
            node = point_load.get_node_applied_to()
            load_vector = point_load.get_load_vector()
            for i in range(self.NUM_DOF_PER_NODE):
                global_dof_num  = identification_array_converted[i, node.get_node_number() - 1]
                global_F[global_dof_num - 1] += load_vector[i]
        
        print("Assembled global F = ")
        print(global_F)
        
        # Solve for displacements
        global_D = np.linalg.solve(global_K, global_F)
        print("Global displacements = ")
        print(global_D)
        return 

    
    # 2D Array where 1 = constrained DOF, 0 = unconstrained DOF
    # Ex.
    #      ID array 
    #    [0, 0, 0, 0]
    #    [0, 1, 0, 1]
    #    [0, 1, 1, 1]
    #
    def create_identification_array(self):
        constrained_dof = np.zeros((3, self.number_of_nodes), dtype=np.int64)
        for j, node in enumerate(self.node_list):
            node_dof = node.get_dof_boundary_conditions()
            for i in range(self.NUM_DOF_PER_NODE):
                constrained_dof[i, j] = node_dof[i]

        return constrained_dof

    # Convert ID array so that 0 = constrained DOF, otherwise number
    # indicates global DOF number
    # Ex.
    #        ID array           Converted ID array
    #      [0, 0, 0, 0]            [1, 4, 5, 7]
    #      [0, 1, 0, 1]    --->    [2, 0, 6, 0]
    #      [0, 1, 1, 1]            [3, 0, 0, 0]
    #
    def convert_identification_array_to_dof_numbering_array(self, identification_array):
        num_unconstrained_dof = 0
        for j in range(len(identification_array)):
            for i in range(self.NUM_DOF_PER_NODE):
                if identification_array[i, j] == 0:
                    identification_array[i, j] = (num_unconstrained_dof + 1)
                    num_unconstrained_dof+= 1
                else:
                    identification_array[i, j] = 0
        
        return identification_array, num_unconstrained_dof
    
    
    # Get 1D Array where index number is equal to the element DOF,
    # and array entry 0 = constrained DOF, other number indicates 
    # global DOF of ith element DOF.
    #
    # Ex.
    #       [0, 4, 5, 0, 0, 2]
    #
    def get_connectivity_array(self, element, identification_array_converted):
        connectivity = np.zeros((self.NUM_DOF_PER_NODE * len(element.get_global_nodes())), dtype=np.int64)
        for j, node in enumerate(element.get_global_nodes()):
            for i in range(self.NUM_DOF_PER_NODE):
                connectivity[j * self.NUM_DOF_PER_NODE+ i] = identification_array_converted[i, node.get_node_number()]
        
        return connectivity


class Frame(Structure):
    pass


class Truss(Structure):
    pass 
    
    

