from this import s
import StructuralElementClass as el
import NodeClass as nd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transform


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
        self.global_D = np.zeros((0,))
        self.boundary_conditions = []
        self.fig, self.ax = plt.subplots()

    
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

    def get_nodes(self):
        return self.node_list

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
        self.global_D = np.linalg.solve(global_K, global_F)
        print("Global displacements = ")
        print(self.global_D)
        return self.global_D

    
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

    def plot_deformed_shape(self):
        if self.global_D.size == 0:
            try:
                raise RuntimeError()
            except:
                print("Displacements of structure have not been solved for yet!")
    
    def plot_structure(self):

        self.ax.axis("equal")

        for node in self.node_list:
            coords = node.get_coordinates()
            self.ax.scatter(coords[0], coords[1], color='black', zorder=1)

        max_x_coord = max(self.ax.get_xlim())
        min_x_coord = min(self.ax.get_xlim())
        max_y_coord = max(self.ax.get_ylim())
        min_y_coord = min(self.ax.get_ylim())
        screen_size = np.sqrt((max_x_coord - min_x_coord)**2 + (max_y_coord - min_y_coord)**2)
        default_scale_factor = screen_size / 30
        patch_list = self.get_boundary_conditions_to_plot(default_scale_factor)
        for x in patch_list: 
            self.ax.add_patch(x)

        for element in self.element_list:
            el_nodes = element.get_global_nodes()
            node1_coords = el_nodes[0].get_coordinates()
            node2_coords = el_nodes[1].get_coordinates()
            x = [node1_coords[0], node2_coords[0]]
            y = [node1_coords[1], node2_coords[1]]
            self.ax.plot(x, y, color='blue', linestyle='solid', linewidth=2.5, zorder=0)

        plt.show()

    def get_boundary_conditions_to_plot(self, scale_factor):
        ## if boundary conditions list has not been filled yet
        #if len(self.boundary_conditions == 0):
        #    for node in self.node_list:
        #        self.boundary_conditions.append(node.get_dof_boundary_conditions())
        #patch_list = []
        
        patch_list = []
        for node in self.node_list:
            coords = node.get_coordinates()
            bc = node.get_dof_boundary_conditions()

            circle_radius = 0.5*scale_factor
            triang_height = scale_factor
            # Y-axis roller
            if np.array_equal(bc, [1, 0, 0]):
                roller_y = patches.Circle([coords[0] + 0.5*scale_factor, coords[1]], radius=0.5*scale_factor, color='red', zorder=2)
                line = patches.Rectangle([coords[0] + scale_factor + 0.05*scale_factor, coords[1] - scale_factor],\
                   0.05*scale_factor, 2*scale_factor, color='red', zorder=2)
                patch_list.append(roller_y)
                patch_list.append(line)
            
            # X-axis roller
            elif np.array_equal(bc, [0, 1, 0]):
                roller_x = patches.Circle([coords[0], coords[1] - 0.5*scale_factor], radius=0.5*scale_factor, color='red', zorder=2)
                line = patches.Rectangle([coords[0] - scale_factor, coords[1] - scale_factor - 0.05*scale_factor],\
                   2*scale_factor, 0.05*scale_factor, color='red', zorder=2)
                patch_list.append(roller_x)
                patch_list.append(line)

            # rotational spring
            elif np.array_equal(bc, [0, 0, 1]):
                ## TODO - IMPLEMENT CURVED ARROWS
                rot_spring = patches.Arc([coords[0], coords[1]], scale_factor, scale_factor, angle=0.0, theta1=135, theta2=45, zorder=2)
                patch_list.append(rot_spring)
            
            # Pin
            elif np.array_equal(bc, [1, 1, 0]):
                pin = plt.Polygon(
                    [[coords[0], coords[1]],
                    [coords[0] - 0.5*scale_factor, coords[1] - scale_factor],
                    [coords[0] + 0.5*scale_factor, coords[1] - scale_factor]],
                    color='red')
                patch_list.append(pin)
            
            # Y-axis roller + rotational spring
            elif np.array_equal(bc, [1, 0, 1]):
                roller_y = patches.Circle([coords[0] + 0.5*scale_factor, coords[1]], radius=0.5*scale_factor, color='red', zorder=2)
                line = patches.Rectangle([coords[0] + scale_factor + 0.05*scale_factor, coords[1] - scale_factor],\
                   0.05*scale_factor, 2*scale_factor, color='red', zorder=2)
                patch_list.append(roller_y)
                patch_list.append(line)

                rot_spring = patches.Arc([coords[0], coords[1]], scale_factor, scale_factor, angle=0.0, theta1=135, theta2=45, zorder=2)
                patch_list.append(rot_spring)
            
            # X-axis roller + rotational spring
            elif np.array_equal(bc, [0, 1, 1]):
                roller_x = patches.Circle([coords[0], coords[1] - 0.5*scale_factor], radius=0.5*scale_factor, color='red', zorder=2)
                patch_list.append(roller_x)

                rot_spring = patches.Arc([coords[0], coords[1]], scale_factor, scale_factor, angle=0.0, theta1=135, theta2=45, zorder=2)
                patch_list.append(rot_spring)
            
            # fully fixed
            # TODO - rotate so that BC is perp to members framing into node
            elif np.array_equal(bc, [1, 1, 1]):
                pin = plt.Polygon(
                   [[coords[0] - scale_factor, coords[1] - 0.2*scale_factor],
                    [coords[0] - scale_factor, coords[1] + 0.2*scale_factor],
                    [coords[0] + scale_factor, coords[1] + 0.2*scale_factor],
                    [coords[0] + scale_factor, coords[1] - 0.2*scale_factor]],
                    color='red', fill=False, closed=True, hatch="///////")
                patch_list.append(pin)

        return patch_list

class Frame(Structure):
    pass


class Truss(Structure):
    pass 
    
    

