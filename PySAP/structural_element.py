from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, List
import abc

if TYPE_CHECKING:
    from node import Node

import utility as util
import numpy as np


""" The current implementation of this package and all of its finite elements
assumes a 2D structural system with the degrees of freedom (DOF) UX, UY, RZ"""


class StructuralLineElement(abc.ABC):
    """Abstract class for a 1D line element (truss, beam, linear spring, etc.)
    All line elements have exactly two nodes. The element is defined as spanning
    between these two nodes.

    Args:
        nodes (List[Node]): A list containing two nodes between which the element spans
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, nodes: List[Node]):
        try:
            assert len(nodes) == 2
        except AssertionError:
            print("elements must be instantiated with exactly two nodes")
        self.nodes = nodes
        self.element_number = -1
        self.local_dof_deformation = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # [u1, v1, R1, u2, v2, R2]
        self.local_force_vector = np.array(    
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # [u1, v1, R1, u2, v2, R2]

    @property
    def element_vector(self) -> np.ndarray:
        return self.nodes[1].coordinates - self.nodes[0].coordinates

    @property
    def element_length(self) -> float:
        return np.linalg.norm(self.element_vector)

    @property
    def angle_relative_to_global_x(self) -> float:
        """positive = CCW from global X-axis"""
        return np.arctan2(self.element_vector[1], self.element_vector[0])

    @abc.abstractmethod
    def get_element_stiffness_matrix(self, transform_to_global_coordinates) -> np.ndarray:
        k11 = 0
        k12 = 0
        k21 = 0
        k22 = 0
        return np.array([[k11, k12],
                         [k21, k22]])

    def process_element_results(self) -> None:
        """Computes local coordinate system element deformations.

        Local element deformations are computed by rotation solved global
        nodal displacements into the local element system. The global solution
        should be solved before calling this function.

        Returns:
            None
        """
        ## TODO: Could refactor this into an @property that calls the element's nodes
        node_i_deformation = self.nodes[0].dof_deformation
        node_j_deformation = self.nodes[1].dof_deformation

        node_i_dof_deformation = util.get_nodal_dof_rotation_matrix(self.angle_relative_to_global_x).T @ node_i_deformation,
        node_j_dof_deformation = util.get_nodal_dof_rotation_matrix(self.angle_relative_to_global_x).T @ node_j_deformation,

        self.local_dof_deformation = np.concatenate(
            (node_i_dof_deformation, node_j_dof_deformation), axis=None
        )
        self.local_force_vector = self.get_element_stiffness_matrix(transform_to_global_coordinates=False) @ self.local_dof_deformation
        return None


class TrussElement(StructuralLineElement):
    """Class representing a structural truss element.

    Truss elements only have axial stiffness, no bending or lateral stiffness.

    Sources:
        Cook et al., Concepts and Applications of Finite Element Analysis, 4th Ed
        Ch. 2

    Args:
        nodes (List[Node]): Node objects between which the TrussElement spans.
        area (Union[int, float]): Cross-sectional area of element.
        elastic_modulus (Union[int, float]): Modulus of elasticity of element.
    """

    def __init__(
        self,
        nodes: List[Node],
        area: Union[int, float],
        elastic_modulus: Union[int, float],
    ):

        super().__init__(nodes)
        self.area = area
        self.elastic_modulus = elastic_modulus

    def get_element_stiffness_matrix(self, transform_to_global_coordinates: bool = True) -> np.ndarray:
        """Computes the stiffness matrix for the TrussElement
        instance in the global or local element coordinate system.

        Source:
        Cook et al., Concepts and Applications of Finite Element Analysis, 4th Ed.
        Pg. 31

        Matrix is of the form:

              NODE i   |   NODE j
          UX   UY   RZ | UX   UY   RZ
        [k11, k12, k13, k14, k15, k16]
        [k12, k22, k23, k24, k25, k26]
        [k13, k23, k33, k34, k35, k36]
        [k14, k24, k34, k44, k45, k46]
        [k15, k25, k35, k45, k55, k56]
        [k16, k26, k36, k46, k56, k66]

        Alternatively, the stiffness matrix could be explicitly defined herein in terms 
        of the local coordinate system stiffness matrix transformed by the transformation matrix: 
        http://mae.uta.edu/~lawrence/me5310/course_materials/me5310_notes/3_Element_Transformations/3-2_Beam_Transformation/3-2_Beam_Transformation.htm
        
        Args:
            convert_to_global_coordinates(bool): Whether to transform the stiffness matrix from the
            element local coordinate system to the global coordinate system.
        
        Returns:
            np.ndarray: A 6x6 matrix with stiffness coefficients for each
            DOF of the truss element.
        """
        if transform_to_global_coordinates:
            c = np.cos(self.angle_relative_to_global_x)
            s = np.sin(self.angle_relative_to_global_x)
        else:
            c = 1
            s = 0
        k = np.array(
            [
                [c**2, c * s, 0, -(c**2), -c * s, 0],
                [c * s, s**2, 0, -c * s, -(s**2), 0],
                [0, 0, 0, 0, 0, 0],
                [-(c**2), -c * s, 0, c**2, c * s, 0],
                [-c * s, -(s**2), 0, c * s, s**2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return k * self.area * self.elastic_modulus / self.element_length

    def get_axial_force_distribution(self, discretization: int) -> np.ndarray:
        """Computes the axial force distribution along the length
        of the TrussElement member.

        Computed based on the axial deformation. See Cook et al. Eq. 2.9-1, and 2.9-5

        Args:
            discretization (int): Number of evenly-spaced points along the member to
            compute axial force.

        Returns:
            None | np.ndarray: returns an array of axial force at "discretization" number of points along the member.
        """
        return np.full(
            discretization,
            (self.local_dof_deformation[3] - self.local_dof_deformation[0])
            * self.elastic_modulus
            * self.area
            / self.element_length,
        )

    def get_shape_functions(self, discretization: int) -> np.ndarray:
        """Computes the shape functions along the length of the TrussElement.

        Linear interpolation functions are used, since the axial displacement
        can only vary linearly along the length of the element.

        Source:
        Cook et al., Concepts and Applications of Finite Element Analysis, 4th Ed.
        Pg. 85, linear interpolation function.

        Args:
            discretization (int): Number of evenly-spaced points along the TrussElement to
            evaluate the shape functions.

        Returns:
            np.ndarray: Returns an array with the calculated values of the shape functions
            for all six DOF of a TrussElement at each discretized point along the member.
        """
        L = self.element_length
        x = np.linspace(0, L, discretization)
        N1 = 1 - x / L
        N2 = 1 - x / L
        N3 = 0 * x
        N4 = x / L
        N5 = x / L
        N6 = 0 * x
        return np.array([N1, N2, N3, N4, N5, N6])

class BeamElement(StructuralLineElement):
    """Class representing a structural beam element.

    Beam elements have axial and bending stiffness.

    Source:
        Cook et al., Concepts and Applications of Finite Element Analysis, 4th Ed
        Ch. 2

    rgs:
        nodes (List[Node]): Node objects between which the TrussElement spans.
        area (int | float): _description_
        elastic_modulus (int | float): Modulus of elasticity of element.
        moment_of_inertia (int | float): Moment of inertia of element.
    """

    def __init__(
        self,
        nodes: List[Node],
        area: int | float,
        elastic_modulus: int | float,
        moment_of_inertia: int | float,
    ):
        super().__init__(nodes)
        self.area = area
        self.elastic_modulus = elastic_modulus
        self.moment_of_inertia = moment_of_inertia
        self.distributed_load_magnitude = 0.0

    def get_element_stiffness_matrix(self, transform_to_global_coordinates: bool = True) -> np.ndarray:
        """Computes the stiffness matrix of the BeamElement instance in the
        global or local element coordinate system.

        Sources:
        Cook et al., Concepts and Applications of Finite Element Analysis, 4th Ed
        Eq. 2.3-5 and https://community.wvu.edu/~bpbettig/MAE456/Lecture_5_Beam_Elements.pdf

        Matrix is of the form:

              NODE i   |   NODE j
          UX   UY   RZ | UX   UY   RZ
        [k11, k12, k13, k14, k15, k16]
        [k12, k22, k23, k24, k25, k26]
        [k13, k23, k33, k34, k35, k36]
        [k14, k24, k34, k44, k45, k46]
        [k15, k25, k35, k45, k55, k56]
        [k16, k26, k36, k46, k56, k66]
        
        Alternatively, the stiffness matrix could be explicitly defined herein in terms 
        of the local coordinate system stiffness matrix transformed by the transformation matrix: 
        http://mae.uta.edu/~lawrence/me5310/course_materials/me5310_notes/3_Element_Transformations/3-2_Beam_Transformation/3-2_Beam_Transformation.htm
               
        Args:
            convert_to_global_coordinates(bool): Whether to transform the stiffness matrix from the
            element local coordinate system to the global coordinate system.
\
        Returns:
            np.ndarray: A 6x6 matrix with stiffness coefficients for each
            DOF of the truss element.
        """
        if transform_to_global_coordinates:
            c = np.cos(self.angle_relative_to_global_x)
            s = np.sin(self.angle_relative_to_global_x)
        else:
            c = 1
            s = 0
        L = self.element_length  # use local variable for readability
        t1 = self.area * L**2 / self.moment_of_inertia
        s2 = s**2
        c2 = c**2

        k11 = t1 * c2 + 12 * s2
        k12 = (t1 - 12) * c * s
        k22 = t1 * s2 + 12 * c2
        k13 = -6 * L * s
        k23 = 6 * L * c
        k33 = 4 * L**2
        k14 = -(t1 * c2 + 12 * s2)
        k24 = -(t1 - 12) * c * s
        k34 = 6 * L * s
        k44 = t1 * c2 + 12 * s2
        k15 = -(t1 - 12) * c * s
        k25 = -(t1 * s2 + 12 * c2)
        k35 = -6 * L * c
        k45 = (t1 - 12) * c * s
        k55 = t1 * s2 + 12 * c2
        k16 = -6 * L * s
        k26 = 6 * L * c
        k36 = 2 * L**2
        k46 = 6 * L * s
        k56 = -6 * L * c
        k66 = 4 * L**2
        k = np.array(  # NODE i  |   NODE j
            [  # UX,  UY,  RZ,| UX,  UY,  RZ,
                [k11, k12, k13, k14, k15, k16],
                [k12, k22, k23, k24, k25, k26],
                [k13, k23, k33, k34, k35, k36],
                [k14, k24, k34, k44, k45, k46],
                [k15, k25, k35, k45, k55, k56],
                [k16, k26, k36, k46, k56, k66],
            ]
        )

        return k * self.moment_of_inertia * self.elastic_modulus / L**3

    def get_axial_force_distribution(self, discretization: int) -> np.ndarray:
        """Computes the axial force distribution along the length
        of the BeamElement.

        Source:
        Cook et al., Concepts and Applications of Finite Element Analysis, 4th Ed
        Eq. 2.9-5

        Args:
            discretization (int): Number of evenly-spaced points along the member to
            compute axial force.

        Returns:
            np.ndarray: returns an array of axial force at "discretization" number of
            points along the member.
        """
        return np.full(
            discretization,
            (self.local_dof_deformation[3] - self.local_dof_deformation[0])
            * self.elastic_modulus
            * self.area
            / self.element_length,
        )

    def get_bending_moment_distribution(self, discretization: int) -> np.ndarray:
        """Computes the bending moment distribution along the length
        of the BeamElement.

        The moment due to nodal displacements is computed using the nodal displacement field multiplied by
        the shape functions. The element moment field, that is, the moment field between nodes, is computed
        by assuming the same moment field as a fixed-fixed beam. The total moment is the sum of the two.

        Sources:
        FEA Theory: Cook et al., Concepts and Applications of Finite Element Analysis, Pgs. 49-51
        Element moment: NDS Beam Design Formulas with Shear and Moment Diagrams, Pg. 15

        Args:
            discretization (int): Number of evenly-spaced points along the member to
            compute bending moment.

        Returns:
            None | np.ndarray: Returns an array of bending moment at "discretization" number of
            points along the member.
        """
        shape_functions = self.get_shape_functions_2nd_derivative(discretization)
        nodal_moment = (
            self.elastic_modulus
            * self.moment_of_inertia
            * (
                self.local_dof_deformation[1] * shape_functions[1]
                + self.local_dof_deformation[2] * shape_functions[2]
                + self.local_dof_deformation[4] * shape_functions[4]
                + self.local_dof_deformation[5] * shape_functions[5]
            )
        )
        local_x = np.linspace(0, self.element_length, discretization)

        element_moment = (self.distributed_load_magnitude / 12) * (
            6 * self.element_length * local_x - self.element_length**2 - 6 * local_x**2
        )
        return nodal_moment + element_moment

    def get_shear_distribution(self, discretization: int) -> np.ndarray:
        """Computes the shear force distribution along the length
        of the BeamElement.

        The shear due to nodal displacements is computed using the nodal displacement field multiplied by
        the shape functions. The element shear field, that is, the shear field between nodes, is computed
        by assuming the same shear field as a fixed-fixed beam. The total shear is the sum of the two.

        Sources:
        FEA Theory: Cook et al., Concepts and Applications of Finite Element Analysis, Pgs. 49-51
        Element shear: NDS Beam Design Formulas with Shear and Moment Diagrams, Pg. 15

        Args:
            discretization (int): Number of evenly-spaced points along the member to
            compute shear force.

        Returns:
            np.ndarray: returns an array of shear force at "discretization" number of points
            along the member.
        """
        shape_functions = self.get_shape_functions_3rd_derivative(discretization)
        nodal_shear = (
            self.elastic_modulus
            * self.moment_of_inertia
            * (
                self.local_dof_deformation[1] * shape_functions[1]
                + self.local_dof_deformation[2] * shape_functions[2]
                + self.local_dof_deformation[4] * shape_functions[4]
                + self.local_dof_deformation[5] * shape_functions[5]
            )
        )
        local_x = np.linspace(0, self.element_length, discretization)
        element_shear = self.distributed_load_magnitude * (
            self.element_length / 2 - local_x
        )
        return nodal_shear + element_shear

    def get_shape_functions(self, discretization: int) -> np.ndarray:
        """Computes the shape functions along the length of the BeamElement.

        Args:
            discretization (int): Number of evenly-spaced points along the BeamElement to
            evaluate the shape functions.

        Returns:
            np.ndarray: Returns an array with the calculated values of the shape functions
            for all six DOF of a BeamElement at each discretized point along the member.
        """
        L = self.element_length
        x = np.linspace(0, L, discretization)
        N1 = 1 - x / L
        N2 = 1 - 3 * x**2 / L**2 + 2 * x**3 / L**3
        N3 = x - 2 * x**2 / L + x**3 / L**2
        N4 = x / L
        N5 = 3 * x**2 / L**2 - 2 * x**3 / L**3
        N6 = -(x**2) / L + x**3 / L**2
        return np.array([N1, N2, N3, N4, N5, N6])

    def get_shape_functions_2nd_derivative(self, discretization: int) -> np.ndarray:
        """Computes the second derivative of the shape functions along the length of the BeamElement.

        Args:
            discretization (int): Number of evenly-spaced points along the BeamElement to
            evaluate the second derivative of the shape functions.

        Returns:
            np.ndarray: Returns an array with the calculated values of the second derivative of the
            shape functions for all six DOF of a BeamElement at each discretized point along the
            member.
        """
        L = self.element_length
        x = np.linspace(0, L, discretization)
        N1 = x * 0
        N2 = -6 / L**2 + 12 * x / L**3
        N3 = -4 / L + 6 * x / L**2
        N4 = x * 0
        N5 = 6 / L**2 - 12 * x / L**3
        N6 = -2 / L + 6 * x / L**2
        return np.array([N1, N2, N3, N4, N5, N6])

    def get_shape_functions_3rd_derivative(self, discretization: int) -> np.ndarray:
        """Computes the third derivative of the shape functions along the length of the BeamElement.

        Args:
            discretization (int): Number of evenly-spaced points along the BeamElement to
            evaluate the third derivative of the shape functions.

        Returns:
            np.ndarray: Returns an array with the calculated values of the third derivative of the
            shape functions for all six DOF of a BeamElement at each discretized point along the
            member.
        """
        L = self.element_length
        xd = np.full(
            discretization, 1
        )  # dummy array so that each shape function will be an array of shape [disc]
        N1 = xd * 0
        N2 = xd * (12 / L**3)
        N3 = xd * (6 / L**2)
        N4 = xd * 0
        N5 = xd * (-12 / L**3)
        N6 = xd * (6 / L**2)
        return np.array([N1, N2, N3, N4, N5, N6])

    def get_equivalent_nodal_load_vector(self, node: Node) -> np.ndarray:
        """Converts the BeamElement distributed load into equivalent nodal load
        vectors for each of the BeamElement's nodes.

        In a finite element solution, loads can only be applied at nodes. Therefore,
        distributed loads along members must be applied at nodes using equivalent nodal
        loads. In the case of beam elements, these equivalent nodal loads consist of
        lateral and rotational forces applied to each of the element's two nodes.

        Args:
            node (Node): _description_

        Returns:
            np.ndarray: _description_
        """
        ### TODO: Change this to not accepting a node argument, and returning either a 6x1 array or
        ### a 2x3 array with the values for both of the element's nodes
        q = self.distributed_load_magnitude
        L = self.element_length
        if node == self.nodes[0]:
            return np.array([0, -q * L / 2, -q * L**2 / 12])
        else:
            return np.array([0, -q * L / 2, q * L**2 / 12])