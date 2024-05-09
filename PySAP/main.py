import structure as struc
import plotter
import structural_element as el
import load as ld
import numpy as np


def main():

    #### Units: lb, in
    a = struc.Structure("Structure 1")
    # angle = -25
    # angle = np.deg2rad(angle)
    # x = 20*np.cos(angle)
    # y = 20*np.sin(angle)
    # a.add_element([[0, 0], [0.5*x, 0.5*y]], el_type="BEAM")
    # a.add_element([[0.5*x, 0.5*y], [x, y]], el_type="BEAM")

    # a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[0, 0])
    # a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[x, y])
    # a.apply_nodal_load(node=a.get_node_nearest_to_coordinate([0.5*x, 0.5*y]), load_vector=[-1000*y, -1000*x, 0])
    
    a.add_element([[0, 0], [50, 0]], el_type="BEAM")
    a.add_element([[50, 0], [100, 0]], el_type="BEAM")
    a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[0, 0])
    a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[100, 0])
    #a.apply_nodal_load(node=a.get_node_nearest_to_coordinate([50, 0]), load_vector=[0, 100, 0])
    a.apply_distributed_load(element_id=0, load_magnitude=-5)
    a.apply_distributed_load(element_id=1, load_magnitude=5)
    plot = plotter.Plotter(a)
    plot.plot_structure()
    
    disp = a.solve()
    plot.plot_deformed_structure(deformed_scale_factor=5)
    plot.plot_moment_diagram()


main()
