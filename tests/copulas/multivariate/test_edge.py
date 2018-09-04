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
        assert left == 2
        assert right == 4
        expected_result = set([1, 3, 5])
        assert depend_set == expected_result

    def test_sort_edge(self):
        sorted_edges = Edge.sort_edge([self.e2, self.e1])
        assert sorted_edges[0].L == 2

    def test_to_dict(self):
        """To_dict returns a dictionary with the parameters to recreate an edge."""
        # Setup
        edge = Edge(2, 5, 'clayton', 1.5)
        edge.D = [1, 3]
        expected_result = {
            'L': 2,
            'R': 5,
            'name': 'clayton',
            'theta': 1.5,
            'D': [1, 3],
            'U': None,
            'likelihood': None,
            'neighbors': None,
            'parents': None,
            'tau': None
        }

        # Run
        result = edge.to_dict()

        # Check
        assert result == expected_result

    def test_from_dict(self):
        """From_dict sets the dictionary values as instance attributes."""
        # Setup
        parameters = {
            'L': 2,
            'R': 5,
            'name': 'clayton',
            'theta': 1.5,
            'D': [1, 3],
            'U': None,
            'likelihood': None,
            'neighbors': [
                {
                    'L': 3,
                    'R': 4,
                    'name': 'gumbel',
                    'theta': 0.4,
                    'D': [2, 5],
                    'U': None,
                    'likelihood': None,
                    'neighbors': None,
                    'parents': None,
                    'tau': None
                }
            ],
            'parents': None,
            'tau': None
        }

        # Run
        edge = Edge.from_dict(**parameters)

        # Check
        assert edge.L == 2
        assert edge.R == 5
        assert edge.name == 'clayton'
        assert edge.theta == 1.5
        assert edge.D == [1, 3]
        assert not edge.U
        assert not edge.parents
        assert len(edge.neighbors) == 1
        assert edge.neighbors[0].L == 3
        assert edge.neighbors[0].R == 4
        assert edge.neighbors[0].name == 'gumbel'
        assert edge.neighbors[0].D == [2, 5]
        assert not edge.neighbors[0].parents
