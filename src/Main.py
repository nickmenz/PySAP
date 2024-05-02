import structure as struc
import plotter
import structural_element as el
import load as ld
import numpy as np



def main():

    #### Units: lb, in
    a = struc.Structure("Structure 1")
    a.add_element([[0, 0], [0, 100]], el_type="BEAM")
    a.add_element([[0, 100], [50, 100]], el_type="BEAM")
    a.add_element([[50, 100], [50, 0]], el_type="BEAM")
    a.apply_boundary_condition(bc_type="fixed", nearest_coordinates=[0, 0])
    a.apply_boundary_condition(bcd_type="fixed", nearest_coordinates=[50, 0])
    a.apply_nodal_load(node_id=1, load_vector=[100, -1000, 0])
    #a.apply_nodal_load(node_id=1, load_vector=[0, -1000, 0])
    plot = plotter.Plotter(a)
    plot.plot_structure()
    
    disp = a.solve()

    plot.plot_deformed_structure(deformed_scale_factor=5)
    print(a.element_list[1].local_dof_deformation)
    #plot.plot_shear_diagram(discretization=50)
main()
