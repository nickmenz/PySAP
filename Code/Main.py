import StructureClass as struc
import NodeClass as nd
import StructuralElementClass as el
import LoadClass as ld
import numpy as np

global NUM_DOF_PER_NODE
NUM_DOF_PER_NODE = 3

def main():

    elastic_mod = 10000000 # ksi
    truss_area = 0.1 # in^2
    
    a = struc.Structure("Structure 1")
    node_1 = nd.Node([-5, 0], [1, 1])
    node_2 = nd.Node([0, -8.66], [0, 0])
    node_3 = nd.Node([5, 0], [1, 1])
    a.add_node(node_1)
    a.add_node(node_2)
    a.add_node(node_3)
    
    element_1 = el.TrussElement([node_1, node_2], truss_area, elastic_mod)
    element_2 = el.TrussElement([node_2, node_3], truss_area, elastic_mod)

    a.add_element(element_1)
    a.add_element(element_2)

    load_1 = ld.PointLoad(np.array([0, -1732]), node_2)

    node_1.set_dof_boundary_conditions(np.array([1, 1, 0], dtype=np.int8))
    node_2.set_dof_boundary_conditions(np.array([0, 0, 0], dtype=np.int8))
    node_3.set_dof_boundary_conditions(np.array([1, 1, 0], dtype=np.int8))

    a.add_point_load(load_1)

    a.plot_structure()
    disp = a.solve()
    



main()
