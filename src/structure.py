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
        #self.view_scale_factor: int | float
        #self.deformation_scale_factor: Optional[int | float] = 500
        #self.fig, self.ax = plt.subplots()

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

    def apply_boundary_condition(self, bc_type: str, node_id: Optional[int] = None, nearest_coordinates: Optional[List[int | float] | np.ndarray] = None):
        if nearest_coordinates is None:
            if node_id is None:
                raise TypeError("No node id or nearest coordinate was provide to apply boundary conditions to")
            else:
                node = self.node_list[node_id]
        else:
            nearest_node = self.get_node_nearest_to_coordinate(nearest_coordinates)
            if nearest_node is None:
                raise TypeError("Could not find nearest node to apply boundary conditions to")
            else:
                node = nearest_node
      
        match bc_type:
            case "fixed":
                node.dof_boundary_conditions = np.array([1, 1, 1])
            case "pinned":
                node.dof_boundary_conditions = np.array([1, 1, 0])
            case "x_roller":
                node.dof_boundary_conditions = np.array([0, 1, 0])
            case "y_roller":
                node.dof_boundary_conditions = np.array([1, 0, 0])
            case "rot":
                node.dof_boundary_conditions = np.array([0, 0, 1])
            case "x_roller_rot":
                node.dof_boundary_conditions = np.array([0, 1, 1])
            case "y_roller_rot":
                node.dof_boundary_conditions = np.array([1, 0, 1])
            case _:
                print(f"Specified boundary condition type on node {node.id} is not valid. Please select a valid boundary condition type")
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
    
