import MemberClass as member


class Structure:
    

    def __init__(self, name):
        self.name = name
        self.member_list = []
    
    def add_member(self, member):
        self.member_list.append(member)
        return

    def list_members(self):
        for member in self.member_list:
            print(member)
    
    def print_name(self):
        print(self.name)
        return

class Frame(Structure):
    pass


class Truss(Structure):
    pass

