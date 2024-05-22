import structure as struc
import plotter
import structural_element as el
import numpy as np


def main():

    #### Units: lb, in
    a = struc.Structure(default_modulus_of_elasticity=29000, default_element_area=4, default_moment_of_inertia=60)
    # angle = 0
    # angle = np.deg2rad(angle)
    # x = 20*np.cos(angle)k
    # y = 20*np.sin(angle)
    # a.add_element([[0, 0], [0.5*x, 0.5*y]], el_type="BEAM")
    # a.add_element([[0.5*x, 0.5*y], [x, y]], el_type="BEAM")

    # a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[0, 0])
    # a.apply_boundary_condition(bc_type="pinned", nearest_coordinates=[x, y])
    # a.apply_nodal_load(at_coordinate=[0.5*x, 0.5*y], load_vector=[0, 0, 100])
    
    a.add_element([[0, 0], [50, 0]], el_type="BEAM")
    a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[0, 0])
    a.apply_nodal_load(at_coordinate=[50, 0], load_vector=[10, -10, 0])

    plot = plotter.Plotter(a)
    plot.plot_structure(show_node_numbers=True, show_element_numbers=True)
    disp = a.solve()
    plot.plot_deformed_structure(deformed_scale_factor=1)
    print(a.reaction_forces)
    plot.plot_shear_diagram()
    plot.plot_moment_diagram()
    plot.plot_axial_diagram()
    #a.print_diagnostics()


main()
