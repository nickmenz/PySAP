from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, List

if TYPE_CHECKING:
    from .node import Node
    import numpy as np


### TODO - remove and make point load an instance variable of node,
    ## distributed load an instance variable of element

# Abstract class
class Load:
    
    def __init__(self, load):
        self.load = load

class PointLoad(Load):
    
    def __init__(self, 
                 load_vector: Union[List[int], List[float], np.ndarray], 
                 node_applied_to: Node):
        ## TODO: THROW EXCEPTION IF 2D ARRAY NOT PASSED IN
        self.load_vector = load_vector
        self.node_applied_to = node_applied_to

class DistributedLoad(Load):
    # For this implementation, it is assumed that the distributed loads
    # always act normal to the member
    def __init__(self, distributed_load_magnitude: int, float):
        self.distributed_load_magnitude = distributed_load_magnitude
    



