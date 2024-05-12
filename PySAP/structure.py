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

    @property
    def minmax_nodal_coordinates(self) -> Dict[str, float]:
        """Minimum and maximum nodal coordinates in the X and Y directions"""
        all_node_coordinates = np.zeros((len(self.node_list), 2))
        for i, node in enumerate(self.node_list):
            all_node_coordinates[i][0] = node.coordinates[0]
            all_node_coordinates[i][1] = node.coordinates[1]

        return {
            "x_min": np.min(all_node_coordinates[:, 0]),
            "x_max": np.max(all_node_coordinates[:, 0]),
            "y_min": np.min(all_node_coordinates[:, 1]),
            "y_max": np.max(all_node_coordinates[:, 1]),
        }

    @property
    def average_element_length(self) -> float:
        """The average length of all elements in the structure"""
        return sum(element.element_length for element in self.element_list) / len(
            self.element_list
        )

    def _add_node(self, node: Node) -> None:
        """Adds a node to the structure object, if not already part of the structure.

        Args:
            node (Node): Node to be added

        Returns:
            None
        """
        if node in self.node_list:
            print("Node has already been added to structure and was not added")
        else:
            node.node_number = len(self.node_list)
            self.node_list.append(node)
        return None

    def get_node_nearest_to_coordinate(
        self, coordinate: List[int | float] | np.ndarray
    ) -> Node | None:
        """Returns the node object that is located nearest to the specified coordinates.

        Args:
            coordinate (List[int  |  float] | np.ndarray): Coordinates at which to find nearest node

        Returns:
            Node | None: Nearest node if the structure has one or more nodes, none if the
            structure does not have any nodes
        """
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
        """Adds an element to the structure.

        Handles the creation of nodes for the elements. For each end coordinate of the element
        If there is no node near it (within a small tolerance), a node will be created.
        Otherwise, the element end will be attached to the existing nearby node.

        Args:
            new_el_node_coordinates (List[int  |  float] | np.ndarray): Coordinates that define each end of the line element.
            el_type (str): BEAM or TRUSS to create a BeamElement and TrussElement, respectively.
            modulus_of_elasticity (Optional[int  |  float], optional): Modulus of elasticity of element. Defaults to Structure
            class default value of none provided.
            area (Optional[int  |  float], optional): Cross-sectional area of element. Defaults to Structure
            class default value of none provided.
            moment_of_inertia (Optional[int  |  float], optional): Moment of inertia of element. Defaults to Structure
            class default value of none provided.

        Raises:
            ValueError: If specified el_type is not BEAM or TRUSS

        Returns:
            None
        """
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
                self._add_node(new_node)
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

        new_element.element_number = len(self.element_list)
        self.element_list.append(new_element)
        return None

    def apply_nodal_load(
        self,
        at_coordinate: List[int | float] | np.ndarray,
        load_vector: List[int | float] | np.ndarray,
    ) -> None:
        """Apply load to node nearest to the specified coordinate.

        Args:
            at_coordinate (List[int  |  float] | np.ndarray): Coordinate at which to search for nearest node to apply load to.
            load_vector (List[int  |  float] | np.ndarray): A 3x1 array giving the nodal load vector [FX, FY, M]

        Raises:
            TypeError: If either at_coordinate or load_vector are not lists or numpy arrays
            AttributeError: If no node nearest to the provided coordinates could be found

        Returns:
            None
        """
        # TODO: Could also add an NSEL type feature to enable adding loads to multiple nodes at once
        if isinstance(at_coordinate, np.ndarray):
            assert (
                at_coordinate.shape[0] == 2 and at_coordinate.size == 2
            ), "at_coordinate array should be of size 2x1"
        elif isinstance(at_coordinate, list):
            assert len(at_coordinate) == 2, "at_coordinate array should be of size 2x1"
        else:
            raise TypeError("at_coordinate must be a numpy array or list")

        node = self.get_node_nearest_to_coordinate(at_coordinate)
        if node is None:
            raise AttributeError(
                "Could not find node nearest to provided coordinates. Make sure at least one element has been added to the structure"
            )

        if isinstance(load_vector, np.ndarray):
            assert (
                load_vector.shape[0] == 3 and load_vector.size == 3
            ), "load_vector array should be of size 3x1"
        elif isinstance(load_vector, list):
            assert len(load_vector) == 3, "load_vector array should be of size 2x1"
        else:
            raise TypeError("load_vector must be a numpy array or list")

        # Cast load vector to np array before assigning to node object attribute
        if not isinstance(load_vector, np.ndarray):
            load_vector = np.array(load_vector)
        node.load_vector = load_vector
        self.nodal_loads[node] = load_vector

        return None

    def apply_distributed_load(
        self, element_id: int, load_magnitude: int | float
    ) -> None:
        """Applies a uniform distributed load to a specified element.

        The load is perpendicular to the element.

        Args:
            element_id (int): Element to apply distributed load to
            load_magnitude (int | float): Magnitude of the uniform distributed load in
            units of force/length

        Returns:
            None
        """
        try:
            element = self.element_list[element_id]
            if isinstance(element, el.BeamElement):
                element.distributed_load_magnitude = load_magnitude
                self.distributed_loads[element] = load_magnitude
            else:
                print(
                    "Distributed load was not applied to element "
                    + str(element_id)
                    + " because distributed loads cannot be applied to truss elements"
                )
        except IndexError:
            print(
                "Element number "
                + str(element_id)
                + " does not exist. \
                  No distributed load has been applied to this element"
            )
        return None

    def apply_boundary_condition(
        self,
        bc_type: str,
        nearest_coordinates: List[int | float] | np.ndarray,
    ):
        """Applies nodal boundary conditions to a node.

        Boundary conditions are applied to the node that is nearest to the coordinates
        specified by nearest_coordinates.

        Args:
            bc_type (str): Type of boundary condition to apply. Must be one of the following: fixed, pinned,
            x_roller (can slide along x-axis but not along y-axis), y_roller, rot, x_roller_rot (x roller plus
            rotational nodal constraint), or y_roller_rot
            nearest_coordinates (List[int  |  float] | np.ndarray): Coordinate at which to search for nearest
            node to apply load to.

        Raises:
            TypeError: If nearest_coordinates is not a list or numpy array
            AttributeError: If no node nearest to the provided coordinates could be found

        Returns:
            None
        """
        if isinstance(nearest_coordinates, np.ndarray):
            assert (
                nearest_coordinates.shape[0] == 2 and nearest_coordinates.size == 2
            ), "nearest_coordinates array should be of size 2x1"
        elif isinstance(nearest_coordinates, list):
            assert (
                len(nearest_coordinates) == 2
            ), "at_coordinate array should be of size 2x1"
        else:
            raise TypeError("at_coordinate must be a numpy array or list")

        nearest_node = self.get_node_nearest_to_coordinate(nearest_coordinates)
        if nearest_node is None:
            raise AttributeError(
                "Could not find nearest node to apply boundary conditions to"
            )
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
                print(
                    f"Specified boundary condition type on node {node.node_number} is not valid. Please select a valid boundary condition type"
                )
        return None

    def renumber_elements_and_nodes(self):
        """Renumbers all of the current elements and nodes
        in the Structure object.

        Returns:
            None
        """
        for i, node in enumerate(self.node_list):
            node.node_number = i

        for i, element in enumerate(self.element_list):
            element.element_number = i

        return None

    def solve(self) -> np.ndarray:
        """Solves KD = F.

        It is first determined whether rotational DOF must be suppressed,
        which is the case if the structure is only comprised of truss elements.
        K and F matrices are assembled and D is solved for. Node object nodal
        displacements are then updated using the global displacements.

        Returns:
            np.ndarray: Global displacement vector where the ith row 
            is the displacement at the ith unconstrained DOF.
        """        
        # Determine whether the structure is solely comprised of truss elements
        # Is structure unstable if rotz not suppressed?
        for element in self.element_list:
            if isinstance(element, el.BeamElement):
                self.is_truss_only_structure = False
                self.suppress_rotz = False
                break

        # Define the ID array where 1 = constrained DOF, 0 = unconstrained DOF
        identification_array = self.create_identification_array()
        print("ID array = ")
        print(identification_array)
        # Convert ID array so that 0 = constrained DOF, other numbers indicate numbering of DOF
        identification_array_converted = (
            self.convert_id_array_to_dof_numbering_array(identification_array)
        )
        print("ID array converted = ")
        print(identification_array_converted)

        global_K = self.assemble_global_k(identification_array_converted)
        global_F = self.assemble_global_f(identification_array_converted)

        # Solve for displacements
        if np.linalg.matrix_rank(global_K) != global_K.shape[0]:
            raise RuntimeError("Global stiffness matrix is singular! Check for an unstable structure or input errors")
        
        self.global_D = np.linalg.solve(global_K, global_F)

        print("Global displacements = ")
        print(self.global_D)

        # Update nodal displacements in node objects
        for node in self.node_list:
            node_dof_deformation = np.array([0.0, 0.0, 0.0])
            node_num = node.node_number
            for i in range(self.NUM_DOF_PER_NODE):
                dof_num = identification_array_converted[i, node_num]
                if dof_num != -1:
                    node_dof_deformation[i] = self.global_D[dof_num]
            node.dof_deformation = node_dof_deformation

        # Update element deformations
        for element in self.element_list:
            element.process_element_results()

        return self.global_D

    def assemble_global_k(self, identification_array_converted: np.ndarray):
        """Assembles the global stiffness matrix for the Structure object.

        The assemblage algorithm detailed in Bathe is used. This involves
        creating a mapping of local element DOF to global DOF numbering
        so that the each coefficient in the local element matrices can be
        readily added to the proper location in the global stiffness matrix.

        An intuitive understanding of the meaning of each coefficient of the 
        global matrix is described in Kassimali: "A structure stiffness coefficient Kij 
        represents the force at the location and in the direction of Pi required, 
        along with other joint forces, to cause a unit value of the displacement dj, 
        while all other joint displacements are zero. Thus, the jth column of the 
        structure stiffness matrix S consists of the joint loads required, at the 
        locations and in the directions of all the degrees of freedom of the structure, 
        to cause a unit value of the displacement dj while all other displacements are zero."

        Sources:
        K.J. Bathe, Finite Element Procedures, 1st Ed.
        Ch 12
        A. Kassimali, Matrix Analysis of Structures, 2nd Ed.
        Sec. 3.7

        Sources:
        K.J. Bathe, Finite Element Procedures, 1st Ed.
        Ch 12
        A. Kassimali, Matrix Analysis of Structures, 2nd Ed.
        Sec. 3.7

        Args:
            identification_array_converted (np.ndarray): An array from the 
            convert_id_array_to_dof_numbering_array that defines the mapping 
            from local element DOF to global DOF numberings.

        Returns:
            np.ndarray: Global stiffness matrix.
        """        
        num_unconstrained_dof = np.max(identification_array_converted) + 1
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
        return global_K

    def assemble_global_f(self, identification_array_converted):
        """Assembles the global force vector of the Structure object.

        Forces due to both nodal loads and distributed loads are applied.
        Distributed loads are applied to the nodes as equivalent nodal loads.
        
        The mapping of local DOF numbering to global DOF numbering is 
        used to add each nodal force to its correct location in the
        global force vector.
        
        Args:
            identification_array_converted (np.ndarray): An array from the 
            convert_id_array_to_dof_numbering_array that defines the mapping 
            from local element DOF to global DOF numberings.

        Returns:
            np.ndarray: Global force vector.
        """        
        num_unconstrained_dof = np.max(identification_array_converted) + 1
        global_F = np.zeros((num_unconstrained_dof))
        for node in self.nodal_loads:
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
                    element.angle_relative_to_global_x
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
        return global_F
    
    def create_identification_array(self) -> np.ndarray:
        """Create array (ID array) that identifies constrained and unconstrained degrees of freedom
        in the structure.

        Returns a 2D array where 1 = constrained DOF, 0 = unconstrained DOF.
        The array is of size DOF x n, where DOF is the number of degrees of freedom
        at each node, and n is the number of nodes in the structure.

        Example:
              ID array
            Node number ->
            [0, 0, 0, 0]
            [0, 1, 0, 1]
            [0, 1, 1, 1]

        Source: K.J. Bathe, Finite Element Procedures, 1st Ed.
        Sec. 12.2.1

        Returns:
            np.ndarray: The ID array.
        """
        constrained_dof = np.zeros((3, len(self.node_list)), dtype=np.int64)
        for j, node in enumerate(self.node_list):
            node_dof = node.dof_boundary_conditions
            for i in range(self.NUM_DOF_PER_NODE):
                constrained_dof[i, j] = node_dof[i]

        if self.suppress_rotz:
            constrained_dof[2, :] = 1

        return constrained_dof

    def convert_id_array_to_dof_numbering_array(
        self, identification_array: np.ndarray
    ) -> np.ndarray:
        """Converts the ID array to an array that defines degree of freedom
        numberings in the structure.

        Constrained DOF are set to a value of -1, unconstrained DOF are
        assigned a global DOF number, starting at 0. The numbering is performed
        by scanning column to column through the ID array and renumbering
        each zero with the DOF number, which increases from 0 to the
        total number of unconstrained DOF minus 1.

        Example:
            4 node structure with 7 unconstrained DOF

            ID array            Converted ID array
            [0, 0, 0, 0]            [0,  3,  4,  6]
            [0, 1, 0, 1]    --->    [1, -1,  5,  0]
            [0, 1, 1, 1]            [2, -1, -1, -1]

        Source: K.J. Bathe, Finite Element Procedures, 1st Ed.
        Sec. 12.2.1

        Args:
            identification_array (np.ndarray): Array that identifies constrained and unconstrained degrees of freedom
        in the structure.

        Returns:
            tuple[np.ndarray, int]: (DOF numbering array, and number of unconstrained DOF)
        """
        num_unconstrained_dof = 0
        for j in range(np.shape(identification_array)[1]):
            for i in range(self.NUM_DOF_PER_NODE):
                if identification_array[i, j] == 0:
                    identification_array[i, j] = num_unconstrained_dof
                    num_unconstrained_dof += 1
                else:
                    identification_array[i, j] = -1

        return identification_array

    def get_connectivity_array(
        self,
        element: BeamElement | TrussElement,
        identification_array_converted: np.ndarray,
    ) -> np.ndarray:
        """Returns array mapping local DOF numbering to global DOF numbering (the connectivity array).

        Local DOF are numbered 0 to 5: [UX1, UY1, RZ1, UX2, UY2, RZ2], where 1 and
        2 are the element's nodes. Global DOF numbering is defined by the
        global DOF numbering array.

        The returned array maps the local DOF numbering to the global. If the DOF
        is constrained, the array entry is equal to -1, otherwise the entry is equal
        to the numbering of the global DOF.

        Example:
            Element with constrained UX1, UX2, UY2:

                    Local DOF #:  0, 1, 2,  3,  4, 5
            connectivity array: [-1, 4, 5, -1, -1, 2]

        Source: K.J. Bathe, Finite Element Procedures, 1st Ed.
        Sec. 12.2.3

        Args:
            element (BeamElement | TrussElement): Element to obtain connectivity array for.
            identification_array_converted (np.ndarray): Array containing global DOF numberings
            for the structure.

        Returns:
            np.ndarray: Connectivity array for the element.
        """
        connectivity = np.zeros(
            (self.NUM_DOF_PER_NODE * len(element.nodes)), dtype=np.int64
        )
        for j, node in enumerate(element.nodes):
            for i in range(self.NUM_DOF_PER_NODE):
                connectivity[j * self.NUM_DOF_PER_NODE + i] = (
                    identification_array_converted[i, node.node_number]
                )

        return connectivity
