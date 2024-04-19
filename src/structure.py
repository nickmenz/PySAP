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
    from matplotlib.figure import Figure
    from matplotlib.axis import Axis
    from matplotlib.patches import Patch


class Structure:
    """_summary_

    Longer summary

    Attributes:
        _type_: _description_
    """
    # This implementation assumes a 2D structural system; i.e., the only DOF
    # are UX, UY, and RZ
    def __init__(
        self,
        name: str = "Structure",
        default_modulus_of_elasticity: int | float = 29000,
        default_element_area: int | float = 3.54,
        default_moment_of_inertia: int | float = 53.8,
    ):
        self.default_modulus_of_elasticity = default_modulus_of_elasticity
        self.default_element_area = default_element_area
        self.default_moment_of_inertia = default_moment_of_inertia
        self.name: str = "unnamed"
        self.node_list: List[Node] = []
        self.element_list: List[TrussElement | BeamElement] = []
        self.nodal_loads: Dict[Node, np.ndarray] = {}
        self.distributed_loads: Dict[BeamElement, int | float] = {}
        self.NUM_DOF_PER_NODE: int = 3
        self.global_D: np.ndarray = np.zeros((0,))
        self.boundary_conditions: List = []
        self.is_truss_only_structure: bool = True
        self.suppress_rotz: bool = True
        self.view_scale_factor: int | float
        self.deformation_scale_factor: Optional[int | float] = 500
        self.fig, self.ax = plt.subplots()

    def add_node(self, node: Node) -> None:
        ## TODO: Reject if node already in list
        self.node_list.append(node)
        return None

    def get_node_nearest_to_coordinate(
        self, coordinate: List[int | float] | np.ndarray
    ) -> Node | None:
        if len(self.node_list) == 0:
            return None
        else:
            existing_coord = np.zeros((len(self.node_list), 2))
            for i, node in enumerate(self.node_list):
                existing_coord[i][0] = node.coordinates[0]
                existing_coord[i][1] = node.coordinates[1]
        # Subtract existing coordinate array from coordinate then take min, compare to val
        distance_to_coord = np.sum((existing_coord - coordinate) ** 2, axis=1)
        nearest_node = self.node_list[np.argmin(distance_to_coord)]
        return nearest_node

    def add_element(
        self,
        new_el_node_coordinates: List[int | float] | np.ndarray,
        el_type: str,
        modulus_of_elasticity: Optional[int | float] = None,
        area: Optional[int | float] = None,
        moment_of_inertia: Optional[int | float] = None,
    ) -> None:
        if modulus_of_elasticity is None:
            E = self.default_modulus_of_elasticity
        else:
            E = modulus_of_elasticity
        if area is None:
            A = self.default_element_area
        else:
            A = area
        if moment_of_inertia is None:
            I = self.default_moment_of_inertia
        else:
            I = moment_of_inertia

        
        ## If there is no node at location, create it. Otherwise, attach element to it
        new_el_node_coordinates = np.array(new_el_node_coordinates)
        new_el_nodes = []
        for i in range(0, new_el_node_coordinates.shape[0]):
            coordinates = new_el_node_coordinates[i]
            nearest_node = self.get_node_nearest_to_coordinate(coordinates)
            if (
                nearest_node is None
                or not np.isclose(
                    coordinates, nearest_node.coordinates, rtol=0.001, atol=0.001
                ).all()
            ):
                new_node = nd.Node(coordinates, np.array([0, 0, 0]))
                self.add_node(new_node)
                new_el_nodes.append(new_node)
            else:
                new_el_nodes.append(nearest_node)

        if el_type.upper() == "BEAM":
            new_element = el.BeamElement(
                new_el_nodes, A, E, I
            )  # type: BeamElement | TrussElement
        elif el_type.upper() == "TRUSS":
            new_element = el.TrussElement(new_el_nodes, A, E)
        else:
            raise ValueError(el_type + " is currently not supported")

        self.element_list.append(new_element)
        return None

    def apply_nodal_load(
        self, node_id: int, load_vector: List[int | float] | np.ndarray
    ) -> None:
        ## TODO: Add option to add nodal loads at a certain coordinate. For the latter case, the node nearest to the
        ## coordinate will be selected for load application

        ## Could also add an NSEL or ESEL type feature to enable adding loads to
        ## multiple nodes at once
        try:
            node = self.node_list[node_id]
            if not isinstance(load_vector, np.ndarray):
                load_vector = np.array(load_vector)
            node.load_vector = load_vector
            self.nodal_loads[node] = load_vector
        except IndexError:
            print(
                "Node number "
                + str(node_id)
                + " does not exist. \
                  No nodal load has been applied to this node"
            )
        return None

    def apply_distributed_load(
        self, element_id: int, load_magnitude: int | float
    ) -> None:
        ## Add ability to apply to a specific element ID. Also add ability to do this
        ## when the element is added to the structure.
        try:
            element = self.element_list[element_id]
            if isinstance(element, el.BeamElement):
                element.distributed_load_magnitude = load_magnitude
                self.distributed_loads[element] = load_magnitude
            else:
                print("Distributed load was not applied to element " + str(element_id) + " because distributed loads cannot be applied to truss elements")
        except IndexError:
            print(
                "Element number "
                + str(element_id)
                + " does not exist. \
                  No distributed load has been applied to this element"
            )
        return None

    def renumber_elements_and_nodes(self):
        # Number all elements and nodes
        for i, node in enumerate(self.node_list):
            node.node_number = i

        for i, element in enumerate(self.element_list):
            element.element_number = i

    def solve(self) -> np.ndarray:        
        for element in self.element_list:
            if isinstance(element, el.BeamElement):
                self.is_truss_only_structure = False
                self.suppress_rotz = False
                break
        
        self.renumber_elements_and_nodes()

        # Define the ID array where 1 = constrained DOF, 0 = unconstrained DOF
        identification_array = self.create_identification_array()
        print("ID array = ")
        print(identification_array)
        # Convert ID array so that 0 = constrained DOF, other numbers indicate numbering of DOF
        identification_array_converted, num_unconstrained_dof = (
            self.convert_id_array_to_dof_numbering_array(identification_array)
        )
        print("ID array converted = ")
        print(identification_array_converted)

        global_K = np.zeros((num_unconstrained_dof, num_unconstrained_dof))
        # Assemble the global stiffness matrix from the individual element stiffness matrices
        for element in self.element_list:
            element_connectivity_array = self.get_connectivity_array(
                element, identification_array_converted
            )
            print(f"Element connectivity array for element {element.element_number} = ")
            print(element_connectivity_array)
            element_k = element.get_element_stiffness_matrix()
            print(f"Element k for element {element.element_number} = ")
            print(element_k)
            for row in range(element_k.shape[0]):
                for col in range(element_k.shape[1]):
                    if (element_connectivity_array[row] != -1) and (
                        element_connectivity_array[col] != -1
                    ):
                        global_K[
                            element_connectivity_array[row],
                            element_connectivity_array[col],
                        ] += element_k[row, col]
        print("Assembled global K = ")
        print(global_K)

        # Assemble the global force vector
        global_F = np.zeros((num_unconstrained_dof))
        print(self.nodal_loads.items())
        for node in self.nodal_loads.keys():
            for i in range(self.NUM_DOF_PER_NODE):
                global_dof_num = identification_array_converted[i, node.node_number]
                # Prevent trying to apply force on constrained DOF
                if global_dof_num == -1 and node.load_vector[i] != 0:
                    print(
                        "Warning - Applied Force has been specified on a constrained DOF. This will be ignored!"
                    )
                else:
                    global_F[global_dof_num] += node.load_vector[i]

        print(self.distributed_loads)
        for element in self.distributed_loads.keys():
            for node in element.nodes:
                equivalent_local_nodal_load_vector = (
                    element.get_equivalent_nodal_load_vector(node)
                )
                trans = util.get_nodal_dof_rotation_matrix(
                    -element.angle_relative_to_global_x.T
                )
                equivalent_global_nodal_load_vector = np.dot(
                    trans, equivalent_local_nodal_load_vector
                )
                for i in range(self.NUM_DOF_PER_NODE):
                    global_dof_num = identification_array_converted[i, node.node_number]
                    # Prevent trying to apply force on constrained DOF
                    if global_dof_num != -1:
                        global_F[global_dof_num] += equivalent_global_nodal_load_vector[
                            i
                        ]

        print("Assembled global F = ")
        print(global_F)

        # Solve for displacements
        self.global_D = np.linalg.solve(global_K, global_F)

        print("Global displacements = ")
        print(self.global_D)

        for node in self.node_list:
            node_dof_deformation = np.array([0.0, 0.0, 0.0])
            node_num = node.node_number
            for i in range(self.NUM_DOF_PER_NODE):
                dof_num = identification_array_converted[i, node_num]
                if dof_num != -1:
                    node_dof_deformation[i] = self.global_D[dof_num]
            node.dof_deformation = node_dof_deformation

        for element in self.element_list:
            element.process_element_results()

        return self.global_D

    # 2D Array where 1 = constrained DOF, 0 = unconstrained DOF
    # Ex.
    #      ID array
    #    [0, 0, 0, 0]
    #    [0, 1, 0, 1]
    #    [0, 1, 1, 1]
    #
    def create_identification_array(self) -> np.ndarray:
        constrained_dof = np.zeros((3, len(self.node_list)), dtype=np.int64)
        for j, node in enumerate(self.node_list):
            node_dof = node.dof_boundary_conditions
            for i in range(self.NUM_DOF_PER_NODE):
                constrained_dof[i, j] = node_dof[i]

        if self.suppress_rotz:
            constrained_dof[2, :] = 1

        return constrained_dof

    # Convert ID array so that -1 = constrained DOF, otherwise number
    # indicates global DOF number (starting at 0)
    # Ex.
    #        ID array           Converted ID array
    #      [0, 0, 0, 0]            [0,  3,  4,  6]
    #      [0, 1, 0, 1]    --->    [1, -1,  5,  0]
    #      [0, 1, 1, 1]            [2, -1, -1, -1]
    #
    def convert_id_array_to_dof_numbering_array(
        self, identification_array: np.ndarray
    ) -> tuple[np.ndarray, int]:
        num_unconstrained_dof = 0
        for j in range(np.shape(identification_array)[1]):
            for i in range(self.NUM_DOF_PER_NODE):
                if identification_array[i, j] == 0:
                    identification_array[i, j] = num_unconstrained_dof
                    num_unconstrained_dof += 1
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
    def get_connectivity_array(
        self, element: BeamElement | TrussElement, identification_array_converted: np.ndarray
    ) -> np.ndarray:
        connectivity = np.zeros(
            (self.NUM_DOF_PER_NODE * len(element.nodes)), dtype=np.int64
        )
        for j, node in enumerate(element.nodes):
            for i in range(self.NUM_DOF_PER_NODE):
                connectivity[j * self.NUM_DOF_PER_NODE + i] = (
                    identification_array_converted[i, node.node_number]
                )

        return connectivity
    
    def initialize_plot(self) -> None:
        self.fig, self.ax = plt.subplots()
        self.ax.axis("equal")
        return None

    def plot_structure(self) -> None:
        if not plt.get_fignums():
            self.initialize_plot()

        for node in self.node_list:
            self.plot_node(node, "undeformed")

        # Get min/max coordinates of structure
        all_node_coordinates = np.zeros((len(self.node_list), 2))
        for i, node in enumerate(self.node_list):
            all_node_coordinates[i][0] = node.coordinates[0]
            all_node_coordinates[i][1] = node.coordinates[1]
        
        min_x_coord = np.min(all_node_coordinates[:,0])
        max_x_coord = np.max(all_node_coordinates[:,0])
        min_y_coord = np.min(all_node_coordinates[:,1])
        max_y_coord = np.max(all_node_coordinates[:,1])
        
        x_center = (max_x_coord - min_x_coord) / 2
        y_center = (max_y_coord - min_y_coord) / 2
        screen_edge_buffer = 1.2
        screen_size = screen_edge_buffer*max(max_x_coord - min_x_coord, max_y_coord -  min_y_coord)
        plt.xlim(x_center - 0.5*screen_size, x_center + 0.5*screen_size)
        plt.ylim(y_center - 0.5*screen_size, y_center + 0.5*screen_size)
        self.view_scale_factor = screen_size / 30

        self.plot_boundary_conditions(self.view_scale_factor)

        for element in self.element_list:
            if isinstance(element, el.TrussElement) or isinstance(
                element, el.BeamElement
            ):
                self.plot_line_element(element, "undeformed")
            else:
                print("Currently, only line elements are supported")

        for node in self.nodal_loads:
            self.plot_nodal_load(node, node.load_vector, self.view_scale_factor)

        return None

    def plot_deformed_structure(self, deformed_scale_factor: int = 1000) -> None:
        if self.global_D.size == 0:
            try:
                raise RuntimeError()
            except RuntimeError:
                print("Displacements of structure have not been solved for yet!")
                return
        self.plot_structure()

        print(f"deformed scale factor = {deformed_scale_factor}")
        if not plt.get_fignums():
            self.initialize_plot()

        for node in self.node_list:
            self.plot_node(node, "deformed", deformed_scale_factor)

        for element in self.element_list:
            if isinstance(element, el.TrussElement) or isinstance(
                element, el.BeamElement
            ):
                self.plot_line_element(element, "deformed", deformed_scale_factor)
            else:
                print("Currently, only line elements are supported")

        plt.show()
        return None

    def plot_node(self, node, display_type, deformed_scale_factor: int = 0) -> None:
        if display_type == "undeformed":
            coords = node.coordinates
            self.ax.scatter(coords[0], coords[1], color="black", zorder=1)
        elif display_type == "deformed":
            coords = (
                node.coordinates
                + node.dof_deformation[0:2] * deformed_scale_factor
            )
            self.ax.scatter(coords[0], coords[1], color="gray", zorder=3)
        else:
            raise RuntimeError("Undefined display type passed to plot_node function")
        return None

    def plot_line_element(
        self,
        element: BeamElement | TrussElement,
        display_type: str,
        deformed_scale_factor: int = 0,
    ) -> None:
        el_nodes = element.nodes
        if display_type == "undeformed":
            node1_coords = el_nodes[0].coordinates
            node2_coords = el_nodes[1].coordinates
            x = np.array([node1_coords[0], node2_coords[0]]) # Using numpy array to avoid type warning
            y = np.array([node1_coords[1], node2_coords[1]]) # Using numpy array to avoid type warning
            self.ax.plot(x, y, color="blue", linestyle="solid", linewidth=2.5, zorder=0)
        elif display_type == "deformed":

            # get shape functions for element and create a discretized local element x-axis
            discretization = 50
            shape_functions = element.get_shape_functions(discretization)
            local_x_axis = np.linspace(0, element.element_length, discretization)

            # get undeformed nodal coordinates and deformations
            node1_x = el_nodes[0].coordinates[0]
            node1_y = el_nodes[0].coordinates[1]
            node2_x = el_nodes[1].coordinates[0]
            node2_y = el_nodes[1].coordinates[1]
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
            x = np.zeros([discretization])
            y = np.zeros([discretization])
            for i in range(discretization):
                trans_coords = np.dot(trans_matrix, np.array([local_x[i], local_y[i]]))
                x[i] = trans_coords[0] + node1_x
                y[i] = trans_coords[1] + node1_y

            self.ax.plot(
                x, y, color="lightblue", linestyle="solid", linewidth=2.5, zorder=0
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
        scale_factor: int | float,
    ) -> None:
        nodal_force = nodal_load_vector[0:2]
        nodal_moment = nodal_load_vector[2]
        if np.any(nodal_force):
            node_coord = node.coordinates
            load_magnitude = round(np.linalg.norm(nodal_force), 2)
            arrow_vector = 3*scale_factor*nodal_force / np.linalg.norm(load_magnitude)

            self.ax.annotate(
                str(load_magnitude),
                [node_coord[0], node_coord[1]],
                xytext=[node_coord[0] - arrow_vector[0], node_coord[1] - arrow_vector[1]],
                xycoords="data",
                textcoords="data",
                horizontalalignment="center",
                arrowprops=dict(edgecolor="green", arrowstyle="->", lw=2),
                zorder=4,
            )
        elif np.any(nodal_moment):
            ## TODO: Implement curved arrows
            x = 2
        
        return None

    def plot_boundary_conditions(self, scale_factor: int | float) -> None:

        patch_list = []  # type: List[Patch]
        for node in self.node_list:
            coords = node.coordinates
            bc = node.dof_boundary_conditions

            circle_radius = 0.5 * scale_factor
            triang_height = scale_factor
            # Y-axis roller
            if np.array_equal(bc, [1, 0, 0]):
                roller_y = patches.Circle(
                    (coords[0] + 0.5 * scale_factor, coords[1]),
                    radius=0.5 * scale_factor,
                    color="red",
                    zorder=2,
                )
                line = patches.Rectangle(
                    (
                        coords[0] + scale_factor + 0.05 * scale_factor,
                        coords[1] - scale_factor,
                    ),
                    0.05 * scale_factor,
                    2 * scale_factor,
                    color="red",
                    zorder=2,
                )
                patch_list.append(roller_y)
                patch_list.append(line)

            # X-axis roller
            elif np.array_equal(bc, [0, 1, 0]):
                roller_x = patches.Circle(
                    (coords[0], coords[1] - 0.5 * scale_factor),
                    radius=0.5 * scale_factor,
                    color="red",
                    zorder=2,
                )
                line = patches.Rectangle(
                    (
                        coords[0] - scale_factor,
                        coords[1] - scale_factor - 0.05 * scale_factor,
                    ),
                    2 * scale_factor,
                    0.05 * scale_factor,
                    color="red",
                    zorder=2,
                )
                patch_list.append(roller_x)
                patch_list.append(line)

            # rotational spring
            elif np.array_equal(bc, [0, 0, 1]):
                ## TODO - IMPLEMENT CURVED ARROWS
                rot_spring = patches.Arc(
                    (coords[0], coords[1]),
                    scale_factor,
                    scale_factor,
                    angle=0.0,
                    theta1=135,
                    theta2=45,
                    zorder=2,
                )
                patch_list.append(rot_spring)

            # Pin
            elif np.array_equal(bc, [1, 1, 0]):
                pin = plt.Polygon(
                    [
                        [coords[0], coords[1]],
                        [coords[0] - 0.5 * scale_factor, coords[1] - scale_factor],
                        [coords[0] + 0.5 * scale_factor, coords[1] - scale_factor],
                    ],
                    color="red",
                )
                patch_list.append(pin)

            # Y-axis roller + rotational spring
            elif np.array_equal(bc, [1, 0, 1]):
                roller_y = patches.Circle(
                    (coords[0] + 0.5 * scale_factor, coords[1]),
                    radius=0.5 * scale_factor,
                    color="red",
                    zorder=2,
                )
                line = patches.Rectangle(
                    (
                        coords[0] + scale_factor + 0.05 * scale_factor,
                        coords[1] - scale_factor,
                    ),
                    0.05 * scale_factor,
                    2 * scale_factor,
                    color="red",
                    zorder=2,
                )
                patch_list.append(roller_y)
                patch_list.append(line)

                rot_spring = patches.Arc(
                    (coords[0], coords[1]),
                    scale_factor,
                    scale_factor,
                    angle=0.0,
                    theta1=135,
                    theta2=45,
                    zorder=2,
                )
                patch_list.append(rot_spring)

            # X-axis roller + rotational spring
            elif np.array_equal(bc, [0, 1, 1]):
                roller_x = patches.Circle(
                    (coords[0], coords[1] - 0.5 * scale_factor),
                    radius=0.5 * scale_factor,
                    color="red",
                    zorder=2,
                )
                patch_list.append(roller_x)

                rot_spring = patches.Arc(
                    (coords[0], coords[1]),
                    scale_factor,
                    scale_factor,
                    angle=0.0,
                    theta1=135,
                    theta2=45,
                    zorder=2,
                )
                patch_list.append(rot_spring)

            # fully fixed
            # TODO - rotate so that BC is perp to members framing into node
            elif np.array_equal(bc, [1, 1, 1]):
                pin = plt.Polygon(
                    [
                        [coords[0] - scale_factor, coords[1] - 0.2 * scale_factor],
                        [coords[0] - scale_factor, coords[1] + 0.2 * scale_factor],
                        [coords[0] + scale_factor, coords[1] + 0.2 * scale_factor],
                        [coords[0] + scale_factor, coords[1] - 0.2 * scale_factor],
                    ],
                    color="red",
                    fill=False,
                    closed=True,
                    hatch="///////",
                )
                patch_list.append(pin)

        for x in patch_list:
            self.ax.add_patch(x)

        return None

    def plot_shear_diagram(
            self, discretization: int, view_scale_factor: float = 0.001
        ) -> None:
            if self.global_D.size == 0:
                try:
                    raise RuntimeError()
                except RuntimeError:
                    print(
                        "Structural displacements and forces have not been solved for yet!"
                    )
                    return
            elif self.is_truss_only_structure:
                try:
                    raise RuntimeError()
                except RuntimeError:
                    print("Shear diagrams cannot be plotted for structures comprised only of truss elements")
            
            self.plot_structure()
            for element in self.element_list:
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

                    self.ax.plot(x, y, color="orange", linestyle="solid", linewidth=1, zorder=0)
                    self.ax.plot(
                        start_point_x,
                        start_point_y,
                        color="orange",
                        linestyle="solid",
                        linewidth=1,
                        zorder=0,
                    )
                    self.ax.plot(
                        end_point_x,
                        end_point_y,
                        color="orange",
                        linestyle="solid",
                        linewidth=1,
                        zorder=0,
                    )

            plt.show()
            return None