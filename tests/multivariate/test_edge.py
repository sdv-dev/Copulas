import logging
from unittest import TestCase

from copulas.multivariate.tree import Edge

LOGGER = logging.getLogger(__name__)

class TestEdge(TestCase):
    def setUp(self):
        self.e1 = Edge(2, 5, 'clayton', 1.5)
        self.e1.D = [1, 3]
        self.e2 = Edge(3, 4, 'clayton', 1.5)
        self.e2.D = [1, 5]

    def test_identify_eds(self):
        left, right, depend_set = Edge._identify_eds_ing(self.e1, self.e2)
        self.assertEquals(left, 2)
        self.assertEquals(right, 4)
        expected_result = set([1, 3, 5])
        self.assertEquals(depend_set, expected_result)
