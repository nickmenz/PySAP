
import StructuralElementClass as el
import NodeClass as nd
import numpy as np
import Utility as util
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
        self.point_loads = {}
        self.number_of_distributed_loads = 0
        self.distributed_loads = {}
        self.NUM_DOF_PER_NODE = 3
        self.global_D = np.zeros((0,))
        self.boundary_conditions = []
        self.suppress_rotz = True
        self.default_view_scale_factor = None
        self.default_deformation_scale_factor = None
        self.fig = None
        self.ax = None


    
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

    def get_elements(self):
        return self.element_list

    def add_point_load(self, load):
        self.point_load_list.append(load)
        self.number_of_point_loads += 1
        return

    def list_point_loads(self):
        for point_load in self.point_load_list:
            print(point_load)
        return

    def get_point_loads(self):
        return self.point_load_list

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
            if np.count_nonzero(node.get_nodal_point_load_vector()) != 0:
                self.point_loads[node] = node.get_nodal_point_load_vector()
        
        for i, element in enumerate(self.element_list):
            element.set_element_number(i)
            if not isinstance(element, el.TrussElement):
                self.suppress_rotz = False
            if element.get_distributed_load_magnitude() != 0:
                self.distributed_loads[element] = element.get_distributed_load_magnitude()

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
                    if((element_connectivity_array[row] != -1) and (element_connectivity_array[col] != -1)):              
                        global_K[element_connectivity_array[row], element_connectivity_array[col]] += element_k[row, col]
        print("Assembled global K = ")
        print(global_K)  
        
        # Assemble the global force vector
        global_F = np.zeros((num_unconstrained_dof))
        print(self.point_loads.items())
        for node, load_vector in self.point_loads.items():
            for i in range(2):
                global_dof_num  = identification_array_converted[i, node.get_node_number()]
                # Prevent trying to apply force on constrained DOF
                if global_dof_num == -1 and load_vector[i] != 0:
                    print("Warning - Applied Force has been specified on a constrained DOF. This will be ignored!")
                else:
                    global_F[global_dof_num] += load_vector[i]
        
        print(self.distributed_loads)
        for element, q in self.distributed_loads.items():
            L = element.get_element_length()
            for node in element.get_global_nodes():
                equivalent_local_nodal_load_vector = element.get_equivalent_nodal_load_vector(node)
                trans = util.get_nodal_dof_rotation_matrix(-element.get_angle_relative_to_global_x().T)
                equivalent_global_nodal_load_vector = np.dot(trans, equivalent_local_nodal_load_vector)
                for i in range(3):
                    global_dof_num  = identification_array_converted[i, node.get_node_number()]
                    # Prevent trying to apply force on constrained DOF
                    if global_dof_num != -1:
                        global_F[global_dof_num] += equivalent_global_nodal_load_vector[i]

                    
        
        print("Assembled global F = ")
        print(global_F)
        
        # Solve for displacements
        self.global_D = np.linalg.solve(global_K, global_F)

        print("Global displacements = ")
        print(self.global_D)

        for node in self.node_list:
            node_dof_deformation = np.array([0.0, 0.0, 0.0])
            node_num = node.get_node_number()
            for i in range(3):
                dof_num = identification_array_converted[i, node_num]
                if dof_num != -1:
                    node_dof_deformation[i] = self.global_D[dof_num]
            node.save_dof_deformation(node_dof_deformation)

        for element in self.element_list:
            element.process_element_results()


        return self.global_D

    
    def get_global_displacements(self):
        if self.global_D.size == 0:
            try:
                raise RuntimeError()
            except:
                print("Displacements of structure have not been solved for yet!")
                return None
            else:
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

        if self.suppress_rotz:
            constrained_dof[2,:] = 1

        return constrained_dof

    # Convert ID array so that -1 = constrained DOF, otherwise number
    # indicates global DOF number (starting at 0)
    # Ex.
    #        ID array           Converted ID array
    #      [0, 0, 0, 0]            [0,  3,  4,  6]
    #      [0, 1, 0, 1]    --->    [1, -1,  5,  0]
    #      [0, 1, 1, 1]            [2, -1, -1, -1]
    #
    def convert_identification_array_to_dof_numbering_array(self, identification_array):
        num_unconstrained_dof = 0
        for j in range(np.shape(identification_array)[1]):
            for i in range(self.NUM_DOF_PER_NODE):
                if identification_array[i, j] == 0:
                    identification_array[i, j] = num_unconstrained_dof
                    num_unconstrained_dof+= 1
                else:
                    identification_array[i, j] = -1
        
        return identification_array, num_unconstrained_dof
    
    
    # Get 1D Array where index number is equal to the element DOF,
    # and array entry -1 = constrained DOF, other number indicates 
    # global DOF of ith element DOF.
    #
    # Ex.
    #       [-1, 4, 5, -1, -1, 2]
    #
    def get_connectivity_array(self, element, identification_array_converted):
        connectivity = np.zeros((self.NUM_DOF_PER_NODE * len(element.get_global_nodes())), dtype=np.int64)
        for j, node in enumerate(element.get_global_nodes()):
            for i in range(self.NUM_DOF_PER_NODE):
                connectivity[j * self.NUM_DOF_PER_NODE + i] = identification_array_converted[i, node.get_node_number()]
        
        return connectivity

    def plot_deformed_structure(self, deformed_scale_factor=1000):
        if self.global_D.size == 0:
            try:
                raise RuntimeError()
            except:
                print("Displacements of structure have not been solved for yet!")
                return
        self.plot_structure()

        if not plt.get_fignums():
            self.initialize_plot()
            
        for node in self.node_list:
            self.plot_node(node, "deformed", deformed_scale_factor)     
            
        for element in self.element_list:
            if isinstance(element, el.TrussElement) or isinstance(element, el.BeamElement):
                self.plot_line_element(element, "deformed", deformed_scale_factor)
            else:
                print("Currently, only line elements are supported")
        
        plt.show()

    
    def plot_structure(self):

        if not plt.get_fignums():
            self.initialize_plot()

        for node in self.node_list:
            self.plot_node(node, "undeformed")
            
        max_x_coord = max(self.ax.get_xlim())
        min_x_coord = min(self.ax.get_xlim())
        max_y_coord = max(self.ax.get_ylim())
        min_y_coord = min(self.ax.get_ylim())
        screen_size = np.sqrt((max_x_coord - min_x_coord)**2 + (max_y_coord - min_y_coord)**2)
        self.default_view_scale_factor = screen_size / 30
        
        self.plot_boundary_conditions(self.default_view_scale_factor)

        for element in self.element_list:
            if isinstance(element, el.TrussElement) or isinstance(element, el.BeamElement):
                self.plot_line_element(element, "undeformed")
            else:
                print("Currently, only line elements are supported")
            
        for node, point_load_vector in self.point_loads.items():
            self.plot_point_load(node, point_load_vector, self.default_view_scale_factor)
        
        #plt.show()

    def initialize_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis("equal")

       
    def plot_node(self, node, display_type, deformed_scale_factor=0):
        if display_type == "undeformed":
            coords = node.get_coordinates()
            self.ax.scatter(coords[0], coords[1], color='black', zorder=1)
        elif display_type == "deformed":
            coords = node.get_coordinates() + node.get_dof_deformation()[0:2] * deformed_scale_factor
            self.ax.scatter(coords[0], coords[1], color='gray', zorder=3)
        else:
             raise RuntimeError("Undefined display type passed to plot_node function")

        
    
    def plot_line_element(self, element, display_type, deformed_scale_factor=0):
        el_nodes = element.get_global_nodes()
        if display_type == "undeformed":
            node1_coords = el_nodes[0].get_coordinates()
            node2_coords = el_nodes[1].get_coordinates()
            x = [node1_coords[0], node2_coords[0]]
            y = [node1_coords[1], node2_coords[1]]
            self.ax.plot(x, y, color='blue', linestyle='solid', linewidth=2.5, zorder=0)
        elif display_type == "deformed":
            
            # get shape functions for element and create a discretized local element x-axis
            discretization = 50
            shape_functions = element.get_shape_functions(discretization)
            local_x_axis = np.linspace(0, element.get_element_length(), discretization)

            # get undeformed nodal coordinates and deformations
            node1_x = el_nodes[0].get_coordinates()[0]
            node1_y = el_nodes[0].get_coordinates()[1]
            node2_x = el_nodes[1].get_coordinates()[0]
            node2_y = el_nodes[1].get_coordinates()[1]
            node1_x_def = element.get_dof_deformation()[0] * deformed_scale_factor
            node1_y_def = element.get_dof_deformation()[1] * deformed_scale_factor
            node1_rot_def = element.get_dof_deformation()[2] * deformed_scale_factor
            node2_x_def = element.get_dof_deformation()[3] * deformed_scale_factor
            node2_y_def = element.get_dof_deformation()[4] * deformed_scale_factor
            node2_rot_def = element.get_dof_deformation()[5] * deformed_scale_factor
            
            # local x and y coordinates of deformed beam
            local_x = (node1_x_def * shape_functions[0] + node2_x_def * shape_functions[3]) + local_x_axis
            local_y = node1_y_def * shape_functions[1] + node2_y_def * shape_functions[4] + node1_rot_def * shape_functions[2] + node2_rot_def * shape_functions[5]
            
            # transformation matrix from local to global
            trans_matrix = util.get_2D_rotation_matrix(-element.get_angle_relative_to_global_x()).T
         
            # transforming each point from global to local
            x = np.zeros([discretization])
            y = np.zeros([discretization])
            for i in range(discretization):
                trans_coords = np.dot(trans_matrix, np.array([local_x[i], local_y[i]]))
                x[i] = trans_coords[0] + node1_x
                y[i] = trans_coords[1] + node1_y
            
            self.ax.plot(x, y, color='lightblue', linestyle='solid', linewidth=2.5, zorder=0)
        else:
            raise RuntimeError("Undefined display type passed to plot_line_element function")

        return


    def plot_point_load(self, node, point_load_vector, scale_factor):
        node_coord = node.get_coordinates()
        load_magnitude = str(round(np.linalg.norm(point_load_vector), 2))
        arrow_vector = 3 * scale_factor * point_load_vector / np.linalg.norm(load_magnitude)
            
        self.ax.annotate(load_magnitude, [node_coord[0], node_coord[1]],\
            xytext=[node_coord[0] - arrow_vector[0], node_coord[1] - arrow_vector[1]], \
            xycoords='data', textcoords='data', horizontalalignment='center', arrowprops=dict(edgecolor='green', \
            arrowstyle='->', lw=2), zorder=4)
        
        return

    def plot_boundary_conditions(self, scale_factor):
        ## if boundary conditions list has not been filled yet
        #if len(self.boundary_conditions == 0):
        #    for node in self.node_list:
        #        self.boundary_conditions.append(node.get_dof_boundary_conditions())
        
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
        
        for x in patch_list: 
            self.ax.add_patch(x)

        return 

class Frame(Structure):
    pass


class Truss(Structure):
    pass 
    
    

