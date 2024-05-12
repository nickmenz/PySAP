import unittest
import structure as struc
import plotter
import numpy as np

class CustomAssertions:
    # vx is expected value, vy is value to be compared against expected
    def assertPercentDifferenceWithinTolerance(self, vx, vy, max_percent_error):
        percent_error = 100 * abs((vx - vy) / vx)
        if np.max(percent_error > max_percent_error):
            raise AssertionError(f"Difference between expected and computed value is greater than {max_percent_error}%")

class TestBenchmarkTrussStructure1(unittest.TestCase, CustomAssertions):
    """Tests solver against results of a benchmark truss structure

        Source: Matrix Analysis of Structures, A. Kassimali, 2nd Edition, Pg. 111
        Units: kip, in
        
    Args:
        unittest (TestCase): Base unit test class
        CustomAssertions (CustomAssertions): Class containing custom assertions.
    """ 
    @classmethod
    def setUpClass(cls):
        elastic_mod = 29000 # ksi
        truss_area = 8.0 # in^2
    
        cls.a = struc.Structure("Structure 1")
        cls.a.add_element([[0,   0], [144, 192]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[144, 0], [144, 192]], el_type="TRUSS", area=6.0, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[288, 0], [144, 192]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[0, 0])
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[144, 0])
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[288, 0])
        cls.a.apply_nodal_load(at_coordinate=[144, 192], load_vector=[150, -300, 0])

        cls.disp = cls.a.solve()

    def test_nodes_added(self):
        """Tests that correct number of nodes have been added to structure.
        """
        self.assertEqual(len(self.a.node_list), 4, "Some nodes have not been properly added to structure")

    def test_elements_added(self):
        """Tests that correct number of elements have been added to structure.
        """
        self.assertEqual(len(self.a.element_list), 3, "Some elements have not been properly added to structure")

    def test_nodal_loads_added(self):
        """Tests that correct number of nodal loads have been added to structure.
        """
        self.assertEqual(len(self.a.nodal_loads), 1, "some nodal loads have not been properly added to the structure")

    def test_displacements_within_error_tolerance(self):
        """Test that computed displacements match theoretical values
        """
        disp_benchmark = [0.21552, -0.13995]
        max_percent_error = 1
        self.assertPercentDifferenceWithinTolerance(disp_benchmark, self.disp, max_percent_error)
    
    def test_plot_structure(self):
        """Test that plotting works correctly
        """
        plot = plotter.Plotter(self.a)
        plot.plot_structure()
        plot.plot_deformed_structure(deformed_scale_factor=100)

class TestBenchmarkTrussStructure2(unittest.TestCase, CustomAssertions):
    """Tests solver against results of a benchmark truss structure

    Source: Matrix Analysis of Structures, A. Kassimali, 2nd Edition, Pg. 115
    Units: kN, mm
    
    Args:
        unittest (TestCase): Base unit test class
        CustomAssertions (CustomAssertions): Class containing custom assertions.
    """
    @classmethod
    def setUpClass(cls):
        elastic_mod = 70 # GPa
        truss_area = 4000 # mm^2
    
        cls.a = struc.Structure("Structure 1")
        cls.a.add_element([[0, 0], [0, 8000]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[0, 0], [6000, 8000]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[0, 8000], [10000, 0]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[10000, 0], [6000, 8000]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.add_element([[0, 8000], [6000, 8000]], el_type="TRUSS", area=truss_area, modulus_of_elasticity=elastic_mod)
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[0, 0])
        cls.a.apply_boundary_condition(bc_type="y_roller", nearest_coordinates=[0, 8000])
        cls.a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[10000, 0])
        cls.a.apply_nodal_load(at_coordinate=[0., 8000.], load_vector=[0, -400, 0])
        cls.a.apply_nodal_load(at_coordinate=[6000., 8000.], load_vector=[800, -400, 0])

        cls.disp = cls.a.solve()

    
    def test_nodes_added(self):
        """Tests that correct number of nodes have been added to structure.
        """
        self.assertEqual(len(self.a.node_list), 4, "Some nodes have not been properly added to structure")

    def test_elements_added(self):
        """Tests that correct number of elements have been added to structure.
        """
        self.assertEqual(len(self.a.element_list), 5, "Some elements have not been properly added to structure")

    def test_nodal_loads_added(self):
        """Tests that correct number of nodal loads have been added to structure.
        """
        self.assertEqual(len(self.a.nodal_loads), 2, "some nodal loads have not been properly added to the structure")

    def test_displacements_within_error_tolerance(self):
        """Test that computed displacements match theoretical values
        """
        disp_benchmark = [-9.1884, 12.837, -9.5846]
        max_percent_error = 1
        self.assertPercentDifferenceWithinTolerance(disp_benchmark, self.disp, max_percent_error)

    def test_plot_structure(self):
        """Test that plotting works correctly
        """
        plot = plotter.Plotter(self.a)
        plot.plot_structure()
        plot.plot_deformed_structure(deformed_scale_factor=100)

class TestFixedBeamStructureVerification(unittest.TestCase, CustomAssertions):
    """Tests solver with a fixed-fixed beam
    
    delta_max = PL^3/192EI
    Units: kip, in
    
    Args:
        unittest (TestCase): Base unit test class
        CustomAssertions (CustomAssertions): Class containing custom assertions.
    """    
    @classmethod
    def setUpClass(cls):
        E = 30000000 # psi
        A = 50.65 # in^2
        I = 7892 # in^4
        P = 50000 # lbs
        L = 300 # in 
        cls.expected_delta_max = -P*L**3/(192*E*I)
    
        cls.a = struc.Structure("Structure 1")
        cls.a.add_element([[0., 0.], [150., 0.]], el_type="BEAM", modulus_of_elasticity=E, area=A, moment_of_inertia=I)
        cls.a.add_element([[150., 0.], [300., 0.]], el_type="BEAM", modulus_of_elasticity=E, area=A, moment_of_inertia=I)
        cls.a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[0., 0.])
        cls.a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[300., 0.])
        cls.a.apply_nodal_load(at_coordinate=[150., 0.], load_vector=[0, -50000, 0])
        
        cls.disp = cls.a.solve()

    def test_nodes_added(self):
        """Tests that correct number of nodes have been added to structure.
        """
        self.assertEqual(len(self.a.node_list), 3, "Some nodes have not been properly added to structure")

    def test_elements_added(self):
        """Tests that correct number of elements have been added to structure.
        """
        self.assertEqual(len(self.a.element_list), 2, "Some elements have not been properly added to structure")

    def test_nodal_loads_added(self):
        """Tests that correct number of nodal loads have been added to structure.
        """
        self.assertEqual(len(self.a.nodal_loads), 1, "some nodal loads have not been properly added to the structure")
   
    def test_displacements_within_error_tolerance(self):
        """Test that computed displacements match theoretical values
        """
        disp_benchmark = self.expected_delta_max
        max_percent_error = 1
        self.assertPercentDifferenceWithinTolerance(disp_benchmark, self.a.get_node_nearest_to_coordinate([150., 0.]).dof_deformation[1], max_percent_error)

    def test_plot_structure(self):
        """Test that plotting works correctly
        """
        plot = plotter.Plotter(self.a)
        plot.plot_structure()
        plot.plot_deformed_structure(deformed_scale_factor=100)

class TestBenchmarkBeamStructure1(unittest.TestCase, CustomAssertions):
    """Tests solver benchmark beam structure

    Units: kip, in
    Source: Structural Analysis, R.C. Hibbeler, 8th Edition, Pg. 601-603
    
    Args:
        unittest (TestCase): Base unit test class
        CustomAssertions (CustomAssertions): Class containing custom assertions.
    """    
    @classmethod
    def setUpClass(cls):
        E = 29000 # psi
        A = 10 # in^2
        I = 500 # in^4
    
        cls.a = struc.Structure("Structure 1")
        cls.a.add_element([[0, 0], [0, 240]], el_type="BEAM", modulus_of_elasticity=E, area=A, moment_of_inertia=I)
        cls.a.add_element([[0, 240], [-240, 240]], el_type="BEAM", modulus_of_elasticity=E, area=A, moment_of_inertia=I)
        cls.a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[0, 0])
        cls.a.apply_boundary_condition(bc_type="x_roller", nearest_coordinates=[-240, 240])
        cls.a.apply_nodal_load(at_coordinate=[0, 240], load_vector=[5, 0, 0])
        
        cls.disp = cls.a.solve()

    def test_nodes_added(self):
        """Tests that correct number of nodes have been added to structure.
        """
        self.assertEqual(len(self.a.node_list), 3, "Some nodes have not been properly added to structure")

    def test_elements_added(self):
        """Tests that correct number of elements have been added to structure.
        """
        self.assertEqual(len(self.a.element_list), 2, "Some elements have not been properly added to structure")

    def test_nodal_loads_added(self):
        """Tests that correct number of nodal loads have been added to structure.
        """
        self.assertEqual(len(self.a.nodal_loads), 1, "some nodal loads have not been properly added to the structure")
   
    def test_displacements_within_error_tolerance(self):
        """Test that computed displacements match theoretical values
        """
        d1 = self.a.get_node_nearest_to_coordinate([0, 240]).dof_deformation[0]
        d2 = self.a.get_node_nearest_to_coordinate([0, 240]).dof_deformation[1]
        d3 = self.a.get_node_nearest_to_coordinate([0, 240]).dof_deformation[2]
        d4 = self.a.get_node_nearest_to_coordinate([-240, 240]).dof_deformation[0]
        d5 = self.a.get_node_nearest_to_coordinate([-240, 240]).dof_deformation[2]
        d1_theor = 0.696 # in
        d2_theor = -0.00155 # in
        d3_theor = -0.002488 # rad
        d4_theor = 0.696 # in
        d5_theor = 0.001234 # rad
        max_percent_error = 1
        self.assertPercentDifferenceWithinTolerance(d1_theor, d1, max_percent_error)
        self.assertPercentDifferenceWithinTolerance(d2_theor, d2, max_percent_error)
        self.assertPercentDifferenceWithinTolerance(d3_theor, d3, max_percent_error)
        self.assertPercentDifferenceWithinTolerance(d4_theor, d4, max_percent_error)
        self.assertPercentDifferenceWithinTolerance(d5_theor, d5, max_percent_error)

    def test_plot_structure(self):
        """Test that plotting works correctly
        """
        plot = plotter.Plotter(self.a)
        plot.plot_structure()
        plot.plot_deformed_structure(deformed_scale_factor=20)

class TestBenchmarkBeamStructure2(unittest.TestCase, CustomAssertions):
    """Tests solver benchmark beam structure
  
    Units: kip, in
    Source: Structural Analysis, R.C. Hibbeler, 8th Edition, Pg. 605-607
    
    Args:
        unittest (TestCase): Base unit test class
        CustomAssertions (CustomAssertions): Class containing custom assertions.
    """    
    @classmethod
    def setUpClass(cls):
        E = 29000 # psi
        A = 12 # in^2
        I = 600 # in^4
    
        cls.a = struc.Structure("Structure 1")
        cls.a.add_element([[0, 0], [240, 180]], el_type="BEAM", modulus_of_elasticity=E, area=A, moment_of_inertia=I)
        cls.a.add_element([[240, 180], [480, 180]], el_type="BEAM", modulus_of_elasticity=E, area=A, moment_of_inertia=I)
        cls.a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[0, 0])
        cls.a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[480, 180])
        cls.a.apply_distributed_load(element_id=1, load_magnitude=3/12) # Convert 3 k/ft to k/in
        
        cls.disp = cls.a.solve()

    def test_nodes_added(self):
        """Tests that correct number of nodes have been added to structure.
        """
        self.assertEqual(len(self.a.node_list), 3, "Some nodes have not been properly added to structure")

    def test_elements_added(self):
        """Tests that correct number of elements have been added to structure.
        """
        self.assertEqual(len(self.a.element_list), 2, "Some elements have not been properly added to structure")

    def test_distributed_loads_added(self):
        """Tests that correct number of distributed loads have been added to structure.
        """
        self.assertEqual(len(self.a.distributed_loads), 1, "some distributed loads have not been properly added to the structure")
   
    def test_displacements_within_error_tolerance(self):
        """Test that computed displacements match theoretical values
        """
        d1 = self.a.get_node_nearest_to_coordinate([240, 180]).dof_deformation[0]
        d2 = self.a.get_node_nearest_to_coordinate([240, 180]).dof_deformation[1]
        d3 = self.a.get_node_nearest_to_coordinate([240, 180]).dof_deformation[2]
        d1_theor = 0.0247 # in
        d2_theor = -0.0954 # rad
        d3_theor = -0.00217 # rad
        max_percent_error = 1
        self.assertPercentDifferenceWithinTolerance(d1_theor, d1, max_percent_error)
        self.assertPercentDifferenceWithinTolerance(d2_theor, d2, max_percent_error)
        self.assertPercentDifferenceWithinTolerance(d3_theor, d3, max_percent_error)

    def test_plot_structure(self):
        """Test that plotting works correctly
        """
        plot = plotter.Plotter(self.a)
        plot.plot_structure()
        plot.plot_deformed_structure(deformed_scale_factor=200)

unittest.main()