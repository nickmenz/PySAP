import structure as struc
import structural_element as el
import load as ld
import numpy as np
import matplotlib.pyplot as plt


def main():

    #### Units: lb, in
    a = struc.Structure("Structure 1")
    a.add_element([[0, 0], [1*240/5, 0]], el_type="BEAM")
    a.add_element([[1*240/5, 0], [2*240/5, 0]], el_type="BEAM")
    a.add_element([[2*240/5, 0], [3*240/5, 0]], el_type="BEAM")
    a.add_element([[3*240/5, 0], [4*240/5, 0]], el_type="BEAM")
    a.add_element([[4*240/5, 0], [5*240/5, 0]], el_type="BEAM")
    a.node_list[0].dof_boundary_conditions = np.array([1, 1, 1])
    a.apply_nodal_load(node_id=5, load_vector=[0, -1, 0])
    disp = a.solve()
    print(disp)
    a.plot_deformed_structure(deformed_scale_factor=5)

main()
