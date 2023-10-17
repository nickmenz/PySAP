import unittest
import LoadClass
import NodeClass
import StructuralElementClass
import StructureClass

#class TestGetAreaRectangleWithSetUp(unittest.TestCase):
 
class TestThatStructureAddsNode(unittest.TestCase):
    def runTest(self):
        a = StructureClass.Structure("Structure 1")
        node_1 = NodeClass.Node([-5, 0], [1, 1])
        node_2 = NodeClass.Node([0, -8.66], [0, 0])
        node_3 = NodeClass.Node([5, 0], [1, 1])
        a.add_node(node_1)
        a.add_node(node_2)
        a.add_node(node_3)
        self.assertEqual(len(a.get_nodes()), 3, "Some nodes have not been properly added to structure")

unittest.main()