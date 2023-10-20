class Load:
    # Abstract constructor
    def __init__(self, load):
        self.load = load


class PointLoad(Load):
    
    def __init__(self, load_vector, node_applied_to):
        ## TODO: THROW EXCEPTION IF 2D ARRAY NOT PASSED IN
        self.load_vector = load_vector
        self.node_applied_to = node_applied_to

    def get_node_applied_to(self):
        return self.node_applied_to
        
    def get_load_vector(self):
        return self.load_vector


class DistributedLoad(Load):
    # For this implementation, it is assumed that the distributed loads
    # always act normal to the member
    def __init__(self, distributed_load_magnitude):
        self.distributed_load_magnitude = distributed_load_magnitude
    
    def get_load_magnitude(self):
        return self.distributed_load_magnitude



