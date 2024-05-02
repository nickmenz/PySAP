from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transform
import structural_element as el
import node as nd
import utility as util

if TYPE_CHECKING:
    from structural_element import TrussElement, BeamElement
    from node import Node
    from structure import Structure
    from matplotlib.figure import Figure
    from matplotlib.axis import Axis
    from matplotlib.patches import Patch


class Plotter:
    """Matplotlib object upon which structure, loads, and results will be plotted.


    Attributes:
        structure: structure object to be plotted.
        view_scale_factor: int or float that defines the scaling factor
        of the plotted elements.
        deformation_scale_factor: int or float that defines how much plotted deformations 
        are scaled relative to actual deformations
        fig, ax: Matplotlib figure and axis objects to be plotted on.
    """
    # This implementation assumes a 2D structural system; i.e., the only DOF
    # are UX, UY, and RZ
    def __init__(
        self,
        structure: Structure
    ):
        self.structure = structure
        self.view_scale_factor: int | float
        self.deformation_scale_factor: Optional[int | float] = 500
        self.fig: Figure = None 
        self.ax: Axis = None


    def initialize_plot(self) -> None:
        """Instantiates the matplotib figure and axis and sets basic viewing settings.

        Returns:
            None
        """        
        self.fig, self.ax = plt.subplots()
        #self.ax.axis("equal")
        self.ax.set_aspect("equal", adjustable="datalim")
        return None

    def plot_structure(self, show_plot: bool = True) -> None:
        """Plots undeformed structural elements, nodes, boundary conditions, and loads. 
        Also sets viewing window.

        Args:
            show_plot (bool, optional): Whether or not to display the plot. Defaults to True.

        Returns:
            None
        """       
        if not plt.get_fignums():
            self.initialize_plot()

        for node in self.structure.node_list:
            self.plot_node(node, "undeformed")

        # Get min/max coordinates of structure
        all_node_coordinates = np.zeros((len(self.structure.node_list), 2))
        for i, node in enumerate(self.structure.node_list):
            all_node_coordinates[i][0] = node.coordinates[0]
            all_node_coordinates[i][1] = node.coordinates[1]
        
        min_x_coord = np.min(all_node_coordinates[:,0])
        max_x_coord = np.max(all_node_coordinates[:,0])
        min_y_coord = np.min(all_node_coordinates[:,1])
        max_y_coord = np.max(all_node_coordinates[:,1])
        
        x_center = (max_x_coord + min_x_coord) / 2
        y_center = (max_y_coord + min_y_coord) / 2
        screen_edge_buffer = 1.2
        screen_size = screen_edge_buffer*max(max_x_coord - min_x_coord, max_y_coord -  min_y_coord)
        plt.xlim(x_center - 0.5*screen_size, x_center + 0.5*screen_size)
        plt.ylim(y_center - 0.5*screen_size, y_center + 0.5*screen_size)
        
        ## TODO: Update to based on average element size or similar technique?
        self.view_scale_factor = screen_size / 30

        self.plot_boundary_conditions()

        for element in self.structure.element_list:
            if isinstance(element, el.TrussElement) or isinstance(
                element, el.BeamElement
            ):
                self.plot_line_element(element, "undeformed")
            else:
                print("Currently, only line elements are supported")

        for node in self.structure.nodal_loads:
            self.plot_nodal_load(node, node.load_vector)

        self.ax.set_aspect("equal", adjustable="box")
        plt.grid(visible=True, linestyle="--", linewidth=0.5)
        self.ax.set_axisbelow(True)
        #self.ax.set_aspect("equal", adjustable="datalim")
        
        if show_plot:
            plt.show()
        return None

    def plot_deformed_structure(self, deformed_scale_factor: int = 1000) -> None:
        """Plots structural elements and nodes in their deformed state.

        Args:
            deformed_scale_factor (int, optional): Factor to multiply actual displacements by for viewing. Defaults to 1000.

        Raises:
            RuntimeError: If structural system has not yet been solved.

        Returns:
            None
        """        
        if self.structure.global_D.size == 0:
            try:
                raise RuntimeError()
            except RuntimeError:
                print("Displacements of structure have not been solved for yet!")
                return
        self.plot_structure(show_plot=False)

        for node in self.structure.node_list:
            self.plot_node(node, "deformed", deformed_scale_factor)

        for element in self.structure.element_list:
            if isinstance(element, el.TrussElement) or isinstance(
                element, el.BeamElement
            ):
                self.plot_line_element(element, "deformed", deformed_scale_factor)
            else:
                print("Currently, only line elements are supported")

        plt.show()
        return None

    def plot_node(self, node, display_type, deformed_scale_factor: int = 1000) -> None:
        """Plots a node.

        Args:
            node (Node): Node object to be plotted.
            display_type (str): Wether to plot at undeformed or deformed location. 
            deformed_scale_factor (int, optional):  Factor to multiply actual displacements by for viewing. Defaults to 1000.

        Raises:
            RuntimeError: If display_type is not "undeformed" or "deformed"

        Returns:
            None
        """        
        if display_type == "undeformed":
            coords = node.coordinates
            self.ax.scatter(coords[0], coords[1], color="black", zorder=2)
        elif display_type == "deformed":
            coords = (
                node.coordinates
                + node.dof_deformation[0:2] * deformed_scale_factor
            )
            self.ax.scatter(coords[0], coords[1], color="gray", zorder=2)
        else:
            raise RuntimeError("Undefined display type passed to plot_node function")
        return None

    def plot_line_element(
        self,
        element: BeamElement | TrussElement,
        display_type: str,
        deformed_scale_factor: int = 1000,
    ) -> None:
        """Plots line elements.

        Args:
            element (BeamElement | TrussElement): Object to be plotted.
            display_type (str): Whether to plot undeformed or deformed element.
            deformed_scale_factor (int, optional): Factor to multiply actual displacements by for viewing. Defaults to 1000.

        Raises:
            RuntimeError: If display_type is not "undeformed" or "deformed"

        Returns:
            None
        """        
        
        el_nodes = element.nodes
        if display_type == "undeformed":
            node1_coords = el_nodes[0].coordinates
            node2_coords = el_nodes[1].coordinates
            x = np.array([node1_coords[0], node2_coords[0]]) # Using numpy array to avoid type warning
            y = np.array([node1_coords[1], node2_coords[1]]) # Using numpy array to avoid type warning
            self.ax.plot(x, y, color="blue", linestyle="solid", linewidth=2.5, zorder=1)

        elif display_type == "deformed":
            # get shape functions for element and create a discretized local element x-axis
            discretization = 50
            shape_functions = element.get_shape_functions(discretization)
            local_x_axis = np.linspace(0, element.element_length, discretization)
            print(f"ELEMENT DEFORMATION = {element.local_dof_deformation}")
            # get nodal deformations
            node1_x_def   = element.local_dof_deformation[0]*deformed_scale_factor
            node1_y_def   = element.local_dof_deformation[1]*deformed_scale_factor
            node1_rot_def = element.local_dof_deformation[2]*deformed_scale_factor
            node2_x_def   = element.local_dof_deformation[3]*deformed_scale_factor
            node2_y_def   = element.local_dof_deformation[4]*deformed_scale_factor
            node2_rot_def = element.local_dof_deformation[5]*deformed_scale_factor

            # local x and y coordinates of deformed beam
            local_x = (
                node1_x_def*shape_functions[0] + node2_x_def*shape_functions[3]
            ) + local_x_axis
            local_y = (
                node1_y_def*shape_functions[1]
                + node2_y_def*shape_functions[4]
                + node1_rot_def*shape_functions[2]
                + node2_rot_def*shape_functions[5]
            )
            # transformation matrix from local to global
            trans_matrix = util.get_2D_rotation_matrix(
                -element.angle_relative_to_global_x
            ).T

            # transforming each point from global to local
            node1_x = el_nodes[0].coordinates[0]
            node1_y = el_nodes[0].coordinates[1]
            x = np.zeros([discretization])
            y = np.zeros([discretization])
            for i in range(discretization):
                trans_coords = np.dot(trans_matrix, np.array([local_x[i], local_y[i]]))
                x[i] = trans_coords[0] + node1_x
                y[i] = trans_coords[1] + node1_y

            self.ax.plot(
                x, y, color="lightblue", linestyle="solid", linewidth=2.5, zorder=1
            )
        else:
            raise RuntimeError(
                "Undefined display type passed to plot_line_element function"
            )
        return None

    def plot_nodal_load(
        self,
        node: Node,
        nodal_load_vector: np.ndarray,
    ) -> None:
        """Plots an arrow showing nodal load.

        Args:
            node (Node): Node object that load is applied to.
            nodal_load_vector (np.ndarray): Vector of load.

        Returns:
            None
        """        
        nodal_force = nodal_load_vector[0:2]
        nodal_moment = nodal_load_vector[2]
        if np.any(nodal_force):
            node_coord = node.coordinates
            load_magnitude = round(np.linalg.norm(nodal_force), 2)
            arrow_vector = 3*self.view_scale_factor*nodal_force / np.linalg.norm(load_magnitude)

            self.ax.annotate(
                str(load_magnitude),
                [node_coord[0], node_coord[1]],
                xytext=[node_coord[0] - arrow_vector[0], node_coord[1] - arrow_vector[1]],
                xycoords="data",
                textcoords="data",
                horizontalalignment="center",
                arrowprops=dict(edgecolor="green", arrowstyle="->", lw=2),
                zorder=5,
            )
        elif np.any(nodal_moment):
            ## TODO: Implement curved arrows
            # rot_spring = patches.Arc(
            #         (coordinates[0], coordinates[1]),
            #         scale_factor*2,
            #         scale_factor*2,
            #         angle=0.0,
            #         theta1=130,
            #         theta2=50,
            #         zorder=2,
            #         color="red"
            #     )
            # arrow_x = (scale_factor)*np.cos(np.deg2rad(50))
            # arrow_y = (scale_factor)*np.sin(np.deg2rad(50))
            # arrow_head = patches.RegularPolygon((arrow_x, arrow_y), 3, radius=scale_factor/5, orientation=np.deg2rad(50), zorder=2, color="red")
            
            # self.patch_list.append(rot_spring)
            # self.patch_list.append(arrow_head)
            
            x = 2
        
        return None

    def plot_boundary_conditions(self) -> None:
        """Plots all boundary conditions for the structure.

        Returns:
            None
        """    
        # Creating temporary variable for readability
        for node in self.structure.node_list:
            coordinates = node.coordinates
            bc = node.dof_boundary_conditions
            angles_of_elements_attached_to_node = [x.angle_relative_to_global_x for x in self.structure.element_list if node in x.nodes]
            average_angle = sum(angles_of_elements_attached_to_node)/len(angles_of_elements_attached_to_node)
            
            # Y-axis roller
            if np.array_equal(bc, [1, 0, 0]):
                self.plot_y_axis_roller_boundary_condition(coordinates)

            # X-axis roller
            elif np.array_equal(bc, [0, 1, 0]):
                self.plot_y_axis_roller_boundary_condition(coordinates)

            # rotational spring
            elif np.array_equal(bc, [0, 0, 1]):
                self.plot_rotational_spring_boundary_condition(coordinates)
        
            # Pin
            elif np.array_equal(bc, [1, 1, 0]):
                self.plot_pin_boundary_condition(coordinates)

            # Y-axis roller + rotational spring
            elif np.array_equal(bc, [1, 0, 1]):
                self.plot_y_axis_roller_boundary_condition(coordinates)
                self.plot_rotational_spring_boundary_condition(coordinates)

            # X-axis roller + rotational spring
            elif np.array_equal(bc, [0, 1, 1]):
                self.plot_x_axis_roller_boundary_condition(coordinates)
                self.plot_rotational_spring_boundary_condition(coordinates)

            # fully fixed
            elif np.array_equal(bc, [1, 1, 1]):
                self.plot_fixed_boundary_condition(coordinates, average_angle)
      
        return None

    def plot_x_axis_roller_boundary_condition(self, coordinates: List | np.ndarray):
        """Plots an x-axis roller.

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.

        Returns:
            None.
        """        
        roller_x = patches.Circle(
                    (coordinates[0], coordinates[1] - 0.5*self.view_scale_factor),
                    radius=0.5*self.view_scale_factor,
                    color="red",
                    zorder=3,
                )
        line = patches.Rectangle(
            (
                coordinates[0] - self.view_scale_factor,
                coordinates[1] - self.view_scale_factor - 0.05*self.view_scale_factor,
            ),
            2*self.view_scale_factor,
            0.05*self.view_scale_factor,
            color="red",
            zorder=3,
        )
        self.ax.add_patch(roller_x)
        self.ax.add_patch(line)
        return None

    def plot_y_axis_roller_boundary_condition(self, coordinates: List | np.ndarray):
        """Plots a y-axis roller.

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.

        Returns:
            None.
        """        
        roller_y = patches.Circle(
                (coordinates[0] - 0.5*self.view_scale_factor, coordinates[1]),
                radius=0.5*self.view_scale_factor,
                color="red",
                zorder=3,
            )
        line = patches.Rectangle(
            (
                coordinates[0] - self.view_scale_factor + 0.05*self.view_scale_factor,
                coordinates[1] - self.view_scale_factor,
            ),
            0.05*self.view_scale_factor,
            2*self.view_scale_factor,
            color="red",
            zorder=3,
        )
        self.ax.add_patch(roller_y)
        self.ax.add_patch(line)
        return None

    def plot_pin_boundary_condition(self, coordinates: List | np.ndarray):
        """_summary_

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.

        Returns:
            None.
        """        
        pin = plt.Polygon(
                        [
                            [coordinates[0], coordinates[1]],
                            [coordinates[0] - 0.5*self.view_scale_factor, coordinates[1] - self.view_scale_factor],
                            [coordinates[0] + 0.5*self.view_scale_factor, coordinates[1] - self.view_scale_factor],
                        ],
                        color="red",
                    )
        self.ax.add_patch(pin)
        return None
    
    def plot_rotational_spring_boundary_condition(self, coordinates: List | np.ndarray):
        """_summary_

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.

        Returns:
            None.
        """        
        theta = np.radians(np.linspace(0,360*3,1000))
        radius = self.view_scale_factor*theta*2/50
        x = radius*np.cos(theta) + coordinates[0]
        y = radius*np.sin(theta) + coordinates[1]
        plt.plot(x, y, color="red", linestyle="solid", linewidth=1.0, zorder=3)
        return None
    
    def plot_fixed_boundary_condition(self, coordinates: List | np.ndarray, average_element_angle: int | float):
        """_summary_

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.
            average_element_angle (int | float): average angle (relative to x-axis) of elements connected to node.

        Returns:
            None
        """     
        
        fixed_bc_coordinates = np.array([[-0.2*self.view_scale_factor, -self.view_scale_factor],
                                            [0.2*self.view_scale_factor, -self.view_scale_factor],
                                            [0.2*self.view_scale_factor, self.view_scale_factor],
                                            [-0.2*self.view_scale_factor, self.view_scale_factor]])
        # Orient patch so that long side is perpendicular to the average angle of the elements framing into the node
        # Patch is vertically oriented by default.
        rot = util.get_2D_rotation_matrix(average_element_angle)
        fixed_bc_coordinates =  fixed_bc_coordinates @ rot.T
        # Transforming from center of (0, 0) to coordinate of node
        fixed_bc_coordinates = fixed_bc_coordinates + coordinates
        fixed_bc = plt.Polygon(
            fixed_bc_coordinates,
            color="red",
            fill=False,
            closed=True,
            hatch="///////",
        )
        self.ax.add_patch(fixed_bc)
        return None
    
    def plot_shear_diagram(
            self, discretization: int, view_scale_factor: float = 0.001
        ) -> None:
            if self.structure.global_D.size == 0:
                try:
                    raise RuntimeError()
                except RuntimeError:
                    print(
                        "Structural displacements and forces have not been solved for yet!"
                    )
                    return
            elif self.structure.is_truss_only_structure:
                try:
                    raise RuntimeError()
                except RuntimeError:
                    print("Shear diagrams cannot be plotted for structures comprised only of truss elements")
            
            self.plot_structure()
            for element in self.structure.element_list:
                if isinstance(element, el.BeamElement):
                    el_nodes = element.nodes
                    shear = element.get_shear_distribution(discretization) * view_scale_factor
                    node1_x = el_nodes[0].coordinates[0]
                    node1_y = el_nodes[0].coordinates[1]
                    node2_x = el_nodes[1].coordinates[0]
                    node2_y = el_nodes[1].coordinates[1]
                    trans_matrix = util.get_2D_rotation_matrix(
                        -element.angle_relative_to_global_x
                    ).T
                    x = np.zeros([discretization])
                    y = np.zeros([discretization])
                    local_x = np.linspace(0, element.element_length, discretization)
                    for i in range(discretization):
                        trans_coords = np.dot(trans_matrix, np.array([local_x[i], shear[i]]))
                        x[i] = trans_coords[0] + node1_x
                        y[i] = trans_coords[1] + node1_y

                    start_point_x = np.array([node1_x, x[0]])
                    end_point_x = np.array([node2_x, x[-1]])
                    start_point_y = np.array([node1_y, y[0]])
                    end_point_y = np.array([node2_y, y[-1]])

                    self.ax.plot(x, y, color="orange", linestyle="solid", linewidth=1, zorder=1)
                    self.ax.plot(
                        start_point_x,
                        start_point_y,
                        color="orange",
                        linestyle="solid",
                        linewidth=1,
                        zorder=1,
                    )
                    self.ax.plot(
                        end_point_x,
                        end_point_y,
                        color="orange",
                        linestyle="solid",
                        linewidth=1,
                        zorder=1,
                    )

            plt.show()
            return None