from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np


class Node:
    def __init__(self, coordinates: np.ndarray, dof_boundary_conditions: np.ndarray):
        self.coordinates = np.array(coordinates, dtype=np.float32)
        self.deformed_coordinates = np.array(coordinates)
        self.dof_deformation = np.array([0., 0., 0.])  # UX, UY, RZ
        self.dof_boundary_conditions = np.array(
            dof_boundary_conditions, dtype=np.int8
        )  # 0 if active (unconstrained) DOF, 1 if inactive (constrained)
        self.node_number = -1
        self.load_vector = np.array([0., 0., 0.])  # UX, UY, RZ
