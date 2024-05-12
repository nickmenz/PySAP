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
        structure: Structure,
        deformation_scale_factor: Optional[int | float] = None,
        use_automatic_deformation_scaling: bool = False,
        discretization: Optional[int] = None,
    ):
        self.structure = structure
        self.fig: Figure
        self.ax: Axis
        if deformation_scale_factor is None:
            self.deformation_scale_factor = 500.0
        else:
            self.deformation_scale_factor = deformation_scale_factor
        if discretization is None:
            self.discretization = 50
        else:
            self.discretization = discretization

    @property
    def plot_axis_length(self) -> float:
        """The plot axis length given the min/max nodal coordinates of the structure.

        Assumes the plot is square.

        """
        # Get min/max coordinates of structure
        minmax = self.structure.minmax_nodal_coordinates
        min_x_coord = minmax["x_min"]
        max_x_coord = minmax["x_max"]
        min_y_coord = minmax["y_min"]
        max_y_coord = minmax["y_max"]
        # Compute axis length
        plot_edge_buffer = 1.2
        plot_axis_length = plot_edge_buffer * max(
            max_x_coord - min_x_coord, max_y_coord - min_y_coord
        )
        return plot_axis_length

    @property
    def plot_center(self) -> tuple[float, float]:
        """Coordinates of center of plot. Assumes that plot is square."""
        # Get min/max coordinates of structure
        minmax = self.structure.minmax_nodal_coordinates
        min_x_coord = minmax["x_min"]
        max_x_coord = minmax["x_max"]
        min_y_coord = minmax["y_min"]
        max_y_coord = minmax["y_max"]
        # Compute center of plot
        x_center = (max_x_coord + min_x_coord) / 2
        y_center = (max_y_coord + min_y_coord) / 2
        return x_center, y_center

    @property
    def view_scale_factor(self) -> float:
        """Scaling factor that can be used to scale size of plotted items"""
        return self.plot_axis_length / 30
        # return self.structure.average_element_length / 25

    # @property
    # def automatic_deformation_scale_factor(self) -> float:
    #     return np.max(self.structure.global_D)

    def initialize_plot(self) -> None:
        """Instantiates the matplotlib figure and axis and sets basic viewing settings.

        Returns:
            None
        """
        self.fig, self.ax = plt.subplots()
        # self.ax.axis("equal")
        self.ax.set_aspect("equal", adjustable="datalim")
        return None

    def plot_structure(self, show_plot: bool = True, show_forces: bool = True) -> None:
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

        # Set axis scales of plot
        x_center, y_center = self.plot_center
        plt.xlim(
            x_center - 0.5 * self.plot_axis_length,
            x_center + 0.5 * self.plot_axis_length,
        )
        plt.ylim(
            y_center - 0.5 * self.plot_axis_length,
            y_center + 0.5 * self.plot_axis_length,
        )

        self.plot_boundary_conditions()

        for element in self.structure.element_list:
            if isinstance(element, el.TrussElement) or isinstance(
                element, el.BeamElement
            ):
                self.plot_line_element(element, "undeformed")
            else:
                print("Currently, only line elements are supported")

        if show_forces:
            for node in self.structure.nodal_loads:
                self.plot_nodal_load(node)

            for element in self.structure.distributed_loads:
                self.plot_distributed_load(element)

        self.ax.set_aspect("equal", adjustable="box")
        plt.grid(visible=True, linestyle="--", linewidth=0.5)
        self.ax.set_axisbelow(True)
        # self.ax.set_aspect("equal", adjustable="datalim")

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
                node.coordinates + node.dof_deformation[0:2] * deformed_scale_factor
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
            x = np.array(
                [node1_coords[0], node2_coords[0]]
            )  # Using numpy array to avoid type warning
            y = np.array(
                [node1_coords[1], node2_coords[1]]
            )  # Using numpy array to avoid type warning
            self.ax.plot(x, y, color="blue", linestyle="solid", linewidth=2.5, zorder=1)

        elif display_type == "deformed":
            # get shape functions for element and create a discretized local element x-axis
            shape_functions = element.get_shape_functions(self.discretization)
            local_x_axis = np.linspace(0, element.element_length, self.discretization)
            # get nodal deformations
            node1_x_def = element.local_dof_deformation[0] * deformed_scale_factor
            node1_y_def = element.local_dof_deformation[1] * deformed_scale_factor
            node1_rot_def = element.local_dof_deformation[2] * deformed_scale_factor
            node2_x_def = element.local_dof_deformation[3] * deformed_scale_factor
            node2_y_def = element.local_dof_deformation[4] * deformed_scale_factor
            node2_rot_def = element.local_dof_deformation[5] * deformed_scale_factor

            # local x and y coordinates of deformed beam
            local_x = (
                node1_x_def * shape_functions[0] + node2_x_def * shape_functions[3]
            ) + local_x_axis
            local_y = (
                node1_y_def * shape_functions[1]
                + node2_y_def * shape_functions[4]
                + node1_rot_def * shape_functions[2]
                + node2_rot_def * shape_functions[5]
            )
            # transformation matrix from local to global
            trans_matrix = util.get_2D_rotation_matrix(
                -element.angle_relative_to_global_x
            ).T

            # transforming each point from global to local
            node1_x = el_nodes[0].coordinates[0]
            node1_y = el_nodes[0].coordinates[1]
            x = np.zeros([self.discretization])
            y = np.zeros([self.discretization])
            for i in range(self.discretization):
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
    ) -> None:
        """Plots an arrow showing nodal load.

        Arrows are plotted in the actual direction in which they act,
        and therefore the absolute value of the load is shown in the
        text annotation.

        Args:
            node (Node): Node object that load is applied to.
            nodal_load_vector (np.ndarray): Vector of load.

        Returns:
            None
        """
        x_force = node.load_vector[0]
        y_force = node.load_vector[1]
        nodal_moment = node.load_vector[2]

        if np.abs(x_force) > 0:
            arrow_length = 4 * self.view_scale_factor
            self.ax.annotate(
                str(abs(round(x_force, 2))),
                [node.coordinates[0], node.coordinates[1]],
                xytext=[
                    node.coordinates[0] - np.sign(x_force) * arrow_length,
                    node.coordinates[1],
                ],
                xycoords="data",
                textcoords="data",
                horizontalalignment="center",
                arrowprops=dict(edgecolor="green", arrowstyle="->", lw=2),
                zorder=5,
                color="green",
                va="center",
            )
        if np.abs(y_force) > 0:
            arrow_length = 4 * self.view_scale_factor
            self.ax.annotate(
                str(abs(round(y_force, 2))),
                [node.coordinates[0], node.coordinates[1]],
                xytext=[
                    node.coordinates[0],
                    node.coordinates[1] - np.sign(y_force) * arrow_length,
                ],
                xycoords="data",
                textcoords="data",
                horizontalalignment="center",
                arrowprops=dict(edgecolor="green", arrowstyle="->", lw=2),
                zorder=5,
                color="green",
                ha="center",
            )
        if np.abs(nodal_moment) > 0:
            load_magnitude = round(nodal_moment, 2)
            self.ax.text(
                node.coordinates[0] + 2 * self.view_scale_factor,
                node.coordinates[1] + 2 * self.view_scale_factor,
                str(load_magnitude),
                color="red",
                ha="center",
                va="center",
            )
            rot_spring = patches.Arc(
                (node.coordinates[0], node.coordinates[1]),
                2 * self.view_scale_factor,
                2 * self.view_scale_factor,
                angle=0.0,
                theta1=130,
                theta2=50,
                zorder=2,
                color="red",
            )
            arrow_x = (self.view_scale_factor) * np.cos(np.deg2rad(50))
            arrow_y = (self.view_scale_factor) * np.sin(np.deg2rad(50))
            arrow_head = patches.RegularPolygon(
                (node.coordinates[0] + arrow_x, node.coordinates[1] + arrow_y),
                3,
                radius=2 * self.view_scale_factor / 5,
                orientation=np.deg2rad(50),
                zorder=2,
                color="red",
            )
            self.ax.add_patch(rot_spring)
            self.ax.add_patch(arrow_head)
        return None

    def plot_distributed_load(self, element) -> None:
        # Plot in local coordinate system then transform to global
        if element.distributed_load_magnitude > 0:
            offset_from_element = 2 * self.view_scale_factor
        else:
            offset_from_element = -2 * self.view_scale_factor
        # Plot line above arrows
        top_line = np.array(
            [[0, offset_from_element], [element.element_length, offset_from_element]]
        )
        top_line = (
            top_line @ util.get_2D_rotation_matrix(element.angle_relative_to_global_x).T
        )
        top_line = top_line + element.nodes[0].coordinates
        self.ax.plot(top_line[:, 0], top_line[:, 1], color="orange", lw=2)

        # Plot arrows
        num_arrows = 5
        arrow_x_vals = np.linspace(0, element.element_length, num=num_arrows)
        for i in range(0, num_arrows):
            arrow_vector = np.array(
                [[arrow_x_vals[i], offset_from_element], [arrow_x_vals[i], 0]]
            )
            arrow_vector = (
                arrow_vector
                @ util.get_2D_rotation_matrix(element.angle_relative_to_global_x).T
            )
            arrow_vector = arrow_vector + element.nodes[0].coordinates
            self.ax.add_patch(
                patches.FancyArrowPatch(
                    posA=arrow_vector[0, :],
                    posB=arrow_vector[1, :],
                    color="orange",
                    arrowstyle="simple",
                    mutation_scale=8,
                )
            )

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
            angles_of_elements_attached_to_node = [
                x.angle_relative_to_global_x
                for x in self.structure.element_list
                if node in x.nodes
            ]
            average_angle = sum(angles_of_elements_attached_to_node) / len(
                angles_of_elements_attached_to_node
            )

            # Y-axis roller
            if np.array_equal(bc, [1, 0, 0]):
                self.plot_y_axis_roller_boundary_condition(coordinates)

            # X-axis roller
            elif np.array_equal(bc, [0, 1, 0]):
                self.plot_x_axis_roller_boundary_condition(coordinates)

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
        """Add an x-axis roller patch to plot.

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.

        Returns:
            None.
        """
        roller_x = patches.Circle(
            (coordinates[0], coordinates[1] - 0.5 * self.view_scale_factor),
            radius=0.5 * self.view_scale_factor,
            color="red",
            zorder=3,
        )
        line = patches.Rectangle(
            (
                coordinates[0] - self.view_scale_factor,
                coordinates[1] - self.view_scale_factor - 0.05 * self.view_scale_factor,
            ),
            2 * self.view_scale_factor,
            0.05 * self.view_scale_factor,
            color="red",
            zorder=3,
        )
        self.ax.add_patch(roller_x)
        self.ax.add_patch(line)
        return None

    def plot_y_axis_roller_boundary_condition(self, coordinates: List | np.ndarray):
        """Adds a y-axis roller patch to plot.

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.

        Returns:
            None.
        """
        roller_y = patches.Circle(
            (coordinates[0] - 0.5 * self.view_scale_factor, coordinates[1]),
            radius=0.5 * self.view_scale_factor,
            color="red",
            zorder=3,
        )
        line = patches.Rectangle(
            (
                coordinates[0] - self.view_scale_factor + 0.05 * self.view_scale_factor,
                coordinates[1] - self.view_scale_factor,
            ),
            0.05 * self.view_scale_factor,
            2 * self.view_scale_factor,
            color="red",
            zorder=3,
        )
        self.ax.add_patch(roller_y)
        self.ax.add_patch(line)
        return None

    def plot_pin_boundary_condition(self, coordinates: List | np.ndarray):
        """Adds a pinned boundary condition patch to plot.

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.

        Returns:
            None.
        """
        pin = plt.Polygon(
            [
                [coordinates[0], coordinates[1]],
                [
                    coordinates[0] - 0.5 * self.view_scale_factor,
                    coordinates[1] - self.view_scale_factor,
                ],
                [
                    coordinates[0] + 0.5 * self.view_scale_factor,
                    coordinates[1] - self.view_scale_factor,
                ],
            ],
            color="red",
        )
        self.ax.add_patch(pin)
        return None

    def plot_rotational_spring_boundary_condition(self, coordinates: List | np.ndarray):
        """Adds a rotational spring boundary condition patch to plot.

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.

        Returns:
            None.
        """
        theta = np.radians(np.linspace(0, 360 * 3, 1000))
        radius = self.view_scale_factor * theta * 2 / 50
        x = radius * np.cos(theta) + coordinates[0]
        y = radius * np.sin(theta) + coordinates[1]
        plt.plot(x, y, color="red", linestyle="solid", linewidth=1.0, zorder=3)
        return None

    def plot_fixed_boundary_condition(
        self, coordinates: List | np.ndarray, average_element_angle: int | float
    ):
        """Adds a fixed boundary condition patch to plot

        Args:
            coordinates (List | np.ndarray): Coordinates of the node that the boundary conditions is applied to.
            average_element_angle (int | float): average angle (relative to x-axis) of elements connected to node.

        Returns:
            None
        """

        fixed_bc_coordinates = np.array(
            [
                [-0.2 * self.view_scale_factor, -self.view_scale_factor],
                [0.2 * self.view_scale_factor, -self.view_scale_factor],
                [0.2 * self.view_scale_factor, self.view_scale_factor],
                [-0.2 * self.view_scale_factor, self.view_scale_factor],
            ]
        )
        # Orient patch so that long side is perpendicular to the average angle of the elements framing into the node
        # Patch is vertically oriented by default.
        rot = util.get_2D_rotation_matrix(average_element_angle)
        fixed_bc_coordinates = fixed_bc_coordinates @ rot.T
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

    def plot_axial_diagram(self) -> None:
        """Creates an axial line diagram plots on all elements in the structure.

        Raises:
            RuntimeError: If structural system has not yet been solved

        Returns:
            None
        """
        if self.structure.global_D.size == 0:
            try:
                raise RuntimeError()
            except RuntimeError:
                print(
                    "Structural displacements and forces have not been solved for yet!"
                )
                return

        max_axial = np.max(
            [
                max(np.abs(element.get_axial_force_distribution(self.discretization)))
                for element in self.structure.element_list
            ]
        )
        self.plot_structure(show_plot=False, show_forces=False)
        for element in self.structure.element_list:
            axial = element.get_axial_force_distribution(self.discretization)
            if np.max(np.abs(axial)) != 0:
                self.plot_line_diagram(element, axial, max_axial, "Axial")

        plt.show()
        return None

    def plot_shear_diagram(self) -> None:
        """Creates a shear line diagram plots on all Beam elements in the structure.

        Raises:
            RuntimeError: If structural system has not yet been solved
            RuntimeError: If structure is solely comprised of truss elements

        Returns:
            None
        """
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
                print(
                    "Shear diagrams cannot be plotted for structures comprised only of truss elements"
                )

        max_shear = np.max(
            [
                max(np.abs(element.get_shear_distribution(self.discretization)))
                for element in self.structure.element_list
            ]
        )
        self.plot_structure(show_plot=False, show_forces=False)
        for element in self.structure.element_list:
            if isinstance(element, el.BeamElement):
                shear = element.get_shear_distribution(self.discretization)
                if np.max(np.abs(shear)) != 0:
                    self.plot_line_diagram(element, shear, max_shear, "Shear")

        plt.show()
        return None

    def plot_moment_diagram(self) -> None:
        """Creates a moment line diagram plots on all Beam elements in the structure.

        Raises:
            RuntimeError: If structural system has not yet been solved
            RuntimeError: If structure is solely comprised of truss elements

        Returns:
            None
        """
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
                print(
                    "moment diagrams cannot be plotted for structures comprised only of truss elements"
                )

        max_moment = np.max(
            [
                max(
                    np.abs(element.get_bending_moment_distribution(self.discretization))
                )
                for element in self.structure.element_list
                if isinstance(element, el.BeamElement)
            ]
        )
        self.plot_structure(show_plot=False, show_forces=False)
        for element in self.structure.element_list:
            if isinstance(element, el.BeamElement):
                moment = element.get_bending_moment_distribution(self.discretization)
                if np.max(np.abs(moment)) != 0:
                    self.plot_line_diagram(element, moment, max_moment, "moment")

        plt.show()
        return None

    def plot_line_diagram(
        self,
        element: BeamElement | TrussElement,
        element_result: np.ndarray,
        max_element_result: float,
        legend_label: str,
    ):
        """Plots line diagrams on elements.

        Results to be plotted are passed into the function as an array of points. Since element results
        are computed in the local element coordinate system, they must be transformed
        to the global system for plotting. First, the results are rotated about the
        local element origin based on the element angle, then they are translated by
        a vector from (0, 0) to location of first node in global coordinates.



        Args:
            element (BeamElement | TrussElement): Element for line diagram results to be plotted on.
            element_result (np.ndarray): An m x 2 array of element results (shear, moment, axial, etc.) to be plotted.
            max_element_result (float): Maximum value on the line diagram for any element in the structure, for scaling.
        """

        assert (
            element_result.shape[0] == self.discretization
        ), "Element result array dimensions must match the specified discretization"

        # Saving results in local coordinate system for plotting text annotations
        node_1_result = element_result[0]
        node_2_result = element_result[-1]

        ### Plot line diagram ###

        # Scale result for viewing
        element_result = (
            2 * self.view_scale_factor * (element_result / max_element_result)
        )

        # Rotate coordinates from local to global and then translate from local to global
        trans_matrix = util.get_2D_rotation_matrix(
            -element.angle_relative_to_global_x
        ).T
        x = np.zeros([self.discretization])
        y = np.zeros([self.discretization])
        node1_x = element.nodes[0].coordinates[0]
        node1_y = element.nodes[0].coordinates[1]
        node2_x = element.nodes[1].coordinates[0]
        node2_y = element.nodes[1].coordinates[1]
        local_element_x = np.linspace(0, element.element_length, self.discretization)
        for i in range(self.discretization):
            # Rotate
            rotated_coords = np.dot(
                trans_matrix, np.array([local_element_x[i], element_result[i]])
            )
            # Translate
            x[i] = rotated_coords[0] + node1_x
            y[i] = rotated_coords[1] + node1_y

        # Plot line diagram
        self.ax.plot(
            x,
            y,
            color="orange",
            linestyle="solid",
            linewidth=1,
            zorder=1,
            label=legend_label,
        )

        # Add lines to connect ends of diagram to the element's nodes
        self.ax.plot(
            (node1_x, x[0]),
            (node1_y, y[0]),
            color="orange",
            linestyle="solid",
            linewidth=1,
            zorder=1,
        )
        self.ax.plot(
            (node2_x, x[-1]),
            (node2_y, y[-1]),
            color="orange",
            linestyle="solid",
            linewidth=1,
            zorder=1,
        )

        # Shade area under plot
        element_global_x = np.linspace(
            element.nodes[0].coordinates[0],
            element.nodes[1].coordinates[0],
            self.discretization,
        )
        element_global_y = np.linspace(
            element.nodes[0].coordinates[1],
            element.nodes[1].coordinates[1],
            self.discretization,
        )
        for i in range(element_global_x.shape[0] - 1):
            self.ax.add_patch(
                plt.Polygon(
                    (
                        [element_global_x[i], element_global_y[i]],
                        [x[i], y[i]],
                        [x[i + 1], y[i + 1]],
                        [element_global_x[i + 1], element_global_y[i + 1]],
                    ),
                    color="orange",
                    alpha=0.2,
                    edgecolor="orange",
                )
            )

        ### Plot text annotations ###

        # Determine rotation angle of text at first node
        element_angle_deg = np.rad2deg(element.angle_relative_to_global_x)
        if node_1_result > 0:
            if element_angle_deg >= 0 and element_angle_deg < 180:
                hor_align = "right"
                ver_align = "bottom"
                rot_angle = element_angle_deg - 90
            else:  # Flip text so that it is right side=up
                hor_align = "left"
                ver_align = "top"
                rot_angle = element_angle_deg + 90
        else:
            if element_angle_deg >= 0 and element_angle_deg < 180:
                hor_align = "left"
                ver_align = "bottom"
                rot_angle = element_angle_deg - 90
            else:  # Flip text so that it is right side=up
                hor_align = "right"
                ver_align = "top"
                rot_angle = element_angle_deg + 90

        # Add annotation at first node
        self.ax.annotate(
            str(round(node_1_result, 2)),
            (x[0], y[0]),
            ha=hor_align,
            va=ver_align,
            rotation=rot_angle,
            rotation_mode="anchor",
            fontsize="small",
            color="orange",
        )

        # Determine rotation angle of text at second node
        if node_2_result > 0:
            if element_angle_deg >= 0 and element_angle_deg < 180:
                hor_align = "right"
                ver_align = "top"
                rot_angle = element_angle_deg - 90
            else:  # Flip text so that it is right side=up
                hor_align = "left"
                ver_align = "bottom"
                rot_angle = element_angle_deg + 90
        else:
            if element_angle_deg >= 0 and element_angle_deg < 180:
                hor_align = "left"
                ver_align = "top"
                rot_angle = element_angle_deg - 90
            else:  # Flip text so that it is right side=up
                hor_align = "right"
                ver_align = "bottom"
                rot_angle = element_angle_deg + 90

        # Add annotation at second node
        self.ax.annotate(
            str(round(node_2_result, 2)),
            (x[-1], y[-1]),
            ha=hor_align,
            va=ver_align,
            rotation=rot_angle,
            rotation_mode="anchor",
            fontsize="small",
            color="orange",
        )
