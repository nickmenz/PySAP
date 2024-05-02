import unittest
import load as ld
import node as nd
import structural_element as el
import plotter
import structure as struc
import numpy as np

class CustomAssertions:

    # vx is expected value, vy is value to be compared against expected
    def assertPercentDifferenceWithinTolerance(self, vx, vy, max_percent_error):
        print(vx)
        print(vy)
        percent_error = 100 * abs((vx - vy) / vx)
        if np.max(percent_error > max_percent_error):
            raise AssertionError(f"Difference between expected and computed value is greater than {max_percent_error}%")


class TestBenchmarkTrussStructure1(unittest.TestCase, CustomAssertions):
    
    @classmethod
    def setUpClass(cls):
        #### Benchmark Problem 1
        #### Matrix Analysis of Structures, A. Kassimali, 2nd Edition, Pg. 111
        #### Units: kip, in
        elastic_mod = 29000 # ksi
        truss_area = 8.0 # in^2
    
        cls.a = struc.Structure("Structure 1")
        cls.a.add_element([[0,   0], [144, 192]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[144, 0], [144, 192]], el_type="TRUSS", area=6.0, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[288, 0], [144, 192]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[0, 0])
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[144, 0])
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[288, 0])
        cls.a.apply_nodal_load(node_id=1, load_vector=[150, -300, 0])
   
        cls.disp = cls.a.solve()

        plot = plotter.Plotter(cls.a)
        plot.plot_structure()

        plot.plot_deformed_structure(deformed_scale_factor=100)
        #plot.plot_shear_diagram(discretization=50)


    def test_nodes_added(cls):
        """Tests that correct number of nodes have been added to structure.
        """        
        cls.assertEqual(len(cls.a.node_list), 4, "Some nodes have not been properly added to structure")

    def test_elements_added(cls):
        """Tests that correct number of elements have been added to structure.
        """        
        cls.assertEqual(len(cls.a.element_list), 3, "Some elements have not been properly added to structure")

    def test_nodal_loads_added(cls):
        """Tests that correct number of nodal loads have been added to structure.
        """        
        cls.assertEqual(len(cls.a.nodal_loads), 1, "some nodal loads have not been properly added to the structure")

    def test_displacements_within_error_tolerance(cls):
        """Test that displacements match expected
        """        
        disp_benchmark = [0.21552, -0.13995]
        max_percent_error = 1
        cls.assertPercentDifferenceWithinTolerance(disp_benchmark, cls.disp, max_percent_error)


class TestBenchmarkTrussStructure2(unittest.TestCase, CustomAssertions):
    
    @classmethod
    def setUpClass(cls):
        #### Benchmark Problem 3
        #### Matrix Analysis of Structures, A. Kassimali, 2nd Edition, Pg. 115
        #### Units: kN, mm
        elastic_mod = 70 # GPa
        truss_area = 4000 # mm^2
    
        cls.a = struc.Structure("Structure 1")
        cls.a.add_element([[0, 0], [0, 8000]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[0, 0], [6000, 8000]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[0, 8000], [10000, 0]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[10000, 0], [6000, 8000]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[0, 8000], [6000, 8000]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[0, 0])
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[144, 0])
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[288, 0])
        cls.a.apply_nodal_load(node_id=1, load_vector=[150, -300, 0])
        load_0 = ld.PointLoad(np.array([0, -400]), cls.node_1)
        load_1 = ld.PointLoad(np.array([800, -400]), cls.node_3)
        
        cls.a = struc.Structure("Structure 1")
        cls.node_0 = nd.Node([0., 0.], [1, 1, 0])
        cls.node_1 = nd.Node([0., 8000.], [1, 0, 0])
        cls.node_2 = nd.Node([10000., 0.], [1, 1, 0])
        cls.node_3 = nd.Node([6000., 8000.], [0, 0, 0])

        cls.a.add_node(cls.node_0)
        cls.a.add_node(cls.node_1)
        cls.a.add_node(cls.node_2)
        cls.a.add_node(cls.node_3)

    
        element_0 = el.TrussElement([cls.node_0, cls.node_1], truss_area, elastic_mod)
        element_1 = el.TrussElement([cls.node_0, cls.node_3], truss_area, elastic_mod)
        element_2 = el.TrussElement([cls.node_1, cls.node_2], truss_area, elastic_mod)
        element_3 = el.TrussElement([cls.node_2, cls.node_3], truss_area, elastic_mod)
        element_4 = el.TrussElement([cls.node_1, cls.node_3], truss_area, elastic_mod)
    
        cls.a.add_element(element_0)
        cls.a.add_element(element_1)
        cls.a.add_element(element_2)
        cls.a.add_element(element_3)
        cls.a.add_element(element_4)
        
        load_0 = ld.PointLoad(np.array([0, -400]), cls.node_1)
        load_1 = ld.PointLoad(np.array([800, -400]), cls.node_3)

        cls.a.add_point_load(load_0)
        cls.a.add_point_load(load_1)
    
        cls.disp = cls.a.solve()


    def test_nodes_added(cls):
        cls.assertEqual(len(cls.a.get_nodes()), 4, "Some nodes have not been properly added to structure")

    def test_elements_added(cls):
        cls.assertEqual(len(cls.a.get_elements()), 5, "Some elements have not been properly added to structure")

    def test_point_loads_added(cls):
        cls.assertEqual(len(cls.a.get_point_loads()), 2, "some point loads have not been properly added to the structure")

    def test_displacements_within_error_tolerance(cls):
        disp_benchmark = [-9.1884, 12.837, -9.5846]
        max_percent_error = 1
        cls.assertPercentDifferenceWithinTolerance(disp_benchmark, cls.disp, max_percent_error)
      
    def get_node1_dof_displacements(cls):
        disp_benchmark = [0, -9.1884]
        max_percent_error = 1
        cls.assertPercentDifferenceWithinTolerance(disp_benchmark, cls.node_2.get_dof_deformation(), max_percent_error)


# class TestFixedBeamStructureVerification(unittest.TestCase, CustomAssertions):
    
#     @classmethod
#     def setUpClass(cls):
#         #### Fixed-Fixed Beam
#         #### delta_max = PL^3/192EI
#         elastic_mod = 30000000 # psi
#         beam_area = 50.65 # in^2
#         moment_of_inertia = 7892 # in^4
#         P = 50000 # lbs
#         L = 300 # in 
#         cls.delta_max = -P * L**3 / (192 * elastic_mod * moment_of_inertia)
#         print("Expected delta = " + str(cls.delta_max))
    
#         cls.a = struc.Structure("Structure 1")
#         cls.node_0 = nd.Node([0., 0.], [1, 1, 1])
#         cls.node_1 = nd.Node([150., 0.], [0, 0, 0])
#         cls.node_2 = nd.Node([300., 0.], [1, 1, 1])
        
#         cls.a.add_node(cls.node_0)
#         cls.a.add_node(cls.node_1)
#         cls.a.add_node(cls.node_2)
            
#         element_0 = el.BeamElement([cls.node_0, cls.node_1], beam_area, elastic_mod, moment_of_inertia)
#         element_1 = el.BeamElement([cls.node_1, cls.node_2], beam_area, elastic_mod, moment_of_inertia)
            
#         cls.a.add_element(element_0)
#         cls.a.add_element(element_1)

#         load_0 = ld.PointLoad(np.array([0, -50000]), cls.node_1)

#         cls.a.add_point_load(load_0)
    
#         cls.disp = cls.a.solve()


#     def test_nodes_added(cls):
#         cls.assertEqual(len(cls.a.get_nodes()), 3, "Some nodes have not been properly added to structure")

#     def test_elements_added(cls):
#         cls.assertEqual(len(cls.a.get_elements()), 2, "Some elements have not been properly added to structure")

#     def test_point_loads_added(cls):
#         cls.assertEqual(len(cls.a.get_point_loads()), 1, "some point loads have not been properly added to the structure")
   
#     def test_max_disp_equal_to_theoretical(cls):
#         disp_benchmark = cls.delta_max
#         max_percent_error = 1
#         cls.assertPercentDifferenceWithinTolerance(disp_benchmark, cls.node_1.get_dof_deformation()[1], max_percent_error)

unittest.main()