import StructureClass as struc
import NodeClass as nd
import MemberClass as mb



def main():
    a = struc.Structure("Structure 1")
    a.print_name()
    member = mb.StructuralMember()
    a.add_member(member)
    a.list_members()



main()
