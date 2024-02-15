import StructureClass as struc
import NodeClass as nd
import StructuralElementClass as el
import LoadClass as ld
import numpy as np
import matplotlib.pyplot as plt


def main():

    #elastic_mod = 10000000 # ksi
    #truss_area = 0.1 # in^2
    
    #a = struc.Structure("Structure 1")
    #node_1 = nd.Node([-5, 0], [1, 1])
    #node_2 = nd.Node([0, -8.66], [0, 0])
    #node_3 = nd.Node([5, 0], [1, 1])
    #a.add_node(node_1)
    #a.add_node(node_2)
    #a.add_node(node_3)
    
    #element_1 = el.TrussElement([node_1, node_2], truss_area, elastic_mod)
    #element_2 = el.TrussElement([node_2, node_3], truss_area, elastic_mod)

    #a.add_element(element_1)
    #a.add_element(element_2)

    #load_1 = ld.PointLoad(np.array([0, -1732]), node_2)

    #node_1.set_dof_boundary_conditions(np.array([1, 1, 0], dtype=np.int8))
    #node_2.set_dof_boundary_conditions(np.array([0, 0, 0], dtype=np.int8))
    #node_3.set_dof_boundary_conditions(np.array([1, 1, 0], dtype=np.int8))

    #a.add_point_load(load_1)

    ##a.plot_structure()
    
    #disp = a.solve()
    ##a.plot_structure()
    #a.plot_deformed_structure()

    #### Benchmark Problem 1
    #### Structural Analysis, R.C. Hibbeler, 8th Edition, Pg. 351
    #elastic_mod = 29000 # ksi
    #truss_area = 0.5 # in^2
    
    #a = struc.Structure("Structure 1")
    #node_1 = nd.Node([0., 0.], [1, 1, 0])
    #node_2 = nd.Node([120., 0.], [0, 0, 0])
    #node_3 = nd.Node([120., 120.], [0, 0, 0])
    #node_4 = nd.Node([240., 0.], [0, 0, 0])
    #node_5 = nd.Node([240., 120.], [0, 0, 0])
    #node_6 = nd.Node([360., 0.], [0, 1, 0])
    #a.add_node(node_1)
    #a.add_node(node_2)
    #a.add_node(node_3)
    #a.add_node(node_4)
    #a.add_node(node_5)
    #a.add_node(node_6)
    
    #element_1 = el.TrussElement([node_1, node_2], truss_area, elastic_mod)
    #element_2 = el.TrussElement([node_1, node_3], truss_area, elastic_mod)
    #element_3 = el.TrussElement([node_2, node_3], truss_area, elastic_mod)
    #element_4 = el.TrussElement([node_2, node_4], truss_area, elastic_mod)
    #element_5 = el.TrussElement([node_3, node_4], truss_area, elastic_mod)
    #element_6 = el.TrussElement([node_3, node_5], truss_area, elastic_mod)
    #element_7 = el.TrussElement([node_4, node_5], truss_area, elastic_mod)
    #element_8 = el.TrussElement([node_4, node_6], truss_area, elastic_mod)
    #element_9 = el.TrussElement([node_5, node_6], truss_area, elastic_mod)

    #a.add_element(element_1)
    #a.add_element(element_2)
    #a.add_element(element_3)
    #a.add_element(element_4)
    #a.add_element(element_5)
    #a.add_element(element_6)
    #a.add_element(element_7)
    #a.add_element(element_8)
    #a.add_element(element_9)

    #load_1 = ld.PointLoad(np.array([0, -4]), node_2)
    #load_2 = ld.PointLoad(np.array([0, -4]), node_4)

    #a.add_point_load(load_1)
    #a.add_point_load(load_2)
    
    #disp = a.solve()
    #a.plot_deformed_structure()
    
    ##### Benchmark Problem 2
    ##### Matrix Analysis of Structures, A. Kassimali, 2nd Edition, Pg. 111
    ##### Units: kip, in
    #elastic_mod = 29000 # ksi
    #truss_area = 8.0 # in^2
    
    #a = struc.Structure("Structure 1")
    #node_1 = nd.Node([0., 0.], [1, 1, 0])
    #node_2 = nd.Node([144., 0.], [1, 1, 0])
    #node_3 = nd.Node([288., 0.], [1, 1, 0])
    #node_4 = nd.Node([144., 192.], [0, 0, 0])

    #a.add_node(node_1)
    #a.add_node(node_2)
    #a.add_node(node_3)
    #a.add_node(node_4)

    
    #element_1 = el.TrussElement([node_1, node_4], truss_area, elastic_mod)
    #element_2 = el.TrussElement([node_2, node_4], 6.0, elastic_mod)
    #element_3 = el.TrussElement([node_3, node_4], truss_area, elastic_mod)

    #a.add_element(element_1)
    #a.add_element(element_2)
    #a.add_element(element_3)

    #load_1 = ld.PointLoad(np.array([150, -300]), node_4)

    #a.add_point_load(load_1)
    
    #disp = a.solve()
    #a.plot_deformed_structure()


    ##### Benchmark Problem 3
    ##### Matrix Analysis of Structures, A. Kassimali, 2nd Edition, Pg. 115
    ##### Units: kN, mm
    #elastic_mod = 70 # GPa
    #truss_area = 4000 # mm^2
    
    #a = struc.Structure("Structure 1")
    #node_1 = nd.Node([0., 0.], [1, 1, 0])
    #node_2 = nd.Node([0., 8000.], [1, 0, 0])
    #node_3 = nd.Node([10000., 0.], [1, 1, 0])
    #node_4 = nd.Node([6000., 8000.], [0, 0, 0])

    #a.add_node(node_1)
    #a.add_node(node_2)
    #a.add_node(node_3)
    #a.add_node(node_4)

    
    #element_1 = el.TrussElement([node_1, node_2], truss_area, elastic_mod)
    #element_2 = el.TrussElement([node_1, node_4], truss_area, elastic_mod)
    #element_3 = el.TrussElement([node_2, node_3], truss_area, elastic_mod)
    #element_4 = el.TrussElement([node_3, node_4], truss_area, elastic_mod)
    #element_5 = el.TrussElement([node_2, node_4], truss_area, elastic_mod)
    
    #a.add_element(element_1)
    #a.add_element(element_2)
    #a.add_element(element_3)
    #a.add_element(element_4)
    #a.add_element(element_5)

    #load_1 = ld.PointLoad(np.array([0, -400]), node_2)
    #load_2 = ld.PointLoad(np.array([800, -400]), node_4)

    #a.add_point_load(load_1)
    #a.add_point_load(load_2)
    
    #disp = a.solve()
    #a.plot_deformed_structure()


    #### Units: lb, in
    elastic_mod = 30000000 # psi
    beam_area = 50.65 # in^2
    moment_of_inertia = 7892 # in^4
    P = 50000 # lbs
    L = 300 # in 
    delta_expected = P * L**3 / (192 * elastic_mod * moment_of_inertia)
    print("Expected delta = " + str(delta_expected))
    
    a = struc.Structure("Structure 1")
    node_0 = nd.Node([0., 0.], [1, 1, 1])
    node_1 = nd.Node([0., 240.], [0, 0, 0])
        
    a.add_node(node_0)
    a.add_node(node_1)
            
    element_0 = el.BeamElement([node_0, node_1], beam_area, elastic_mod, moment_of_inertia)
            
    a.add_element(element_0)

    #element_0.apply_distributed_load(500)
    #element_1.apply_distributed_load(500)
    node_1.apply_point_load(np.array([5000, 0, 0]))

    
    disp = a.solve()
    a.plot_deformed_structure()
    disc=50
    a.plot_shear_diagram(disc)


main()
