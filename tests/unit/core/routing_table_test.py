import unittest
from parameterized import parameterized
from unittest.mock import patch, MagicMock, call
from core.routing import RoutingTable, RouteOptimizer

class TestRoutingTable(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.node_pool = [MagicMock(id=f"node_{i:03}") for i in range(100)]
        
    def setUp(self):
        self.topology = MagicMock()
        self.routing_table = RoutingTable(self.topology, cache_size=1000)
        
    @parameterized.expand([
        ("single_node", 1),
        ("scale_100_nodes", 100),
        ("overflow_capacity", 1001)
    ])
    def test_node_registration_scenarios(self, _, node_count):
        test_nodes = self.node_pool[:node_count]
        
        for node in test_nodes:
            self.routing_table.register_node(node)
            
        if node_count <= 1000:
            self.assertEqual(self.routing_table.node_count, node_count)
        else:
            self.assertEqual(self.routing_table.node_count, 1000)
            
    def test_route_calculation_with_failures(self):
        self.topology.find_shortest_path.side_effect = [
            None,
            ["A", "B", "C"],
            None
        ]
        
        with self.subTest("test_no_route_found"):
            self.assertIsNone(self.routing_table.calculate_route("X", "Y"))
            
        with self.subTest("test_valid_route"):
            route = self.routing_table.calculate_route("A", "C")
            self.assertEqual(route.node_sequence, ["A", "B", "C"])
            
    @patch('core.routing.RouteOptimizer.verify_latency')
    def test_route_validation_workflow(self, mock_verify):
        mock_verify.side_effect = [True, False, TimeoutError]
        
        test_cases = [
            (["A", "B"], True),
            (["X", "Y"], False),
            (["M", "N"], False)
        ]
        
        for path, expected in test_cases:
            with self.subTest(f"Testing path {path}"):
                result = self.routing_table.validate_route(path)
                self.assertEqual(result, expected)
                
    def test_cache_invalidation_policy(self):
        test_route = MagicMock()
        test_route.last_used = 0  # Epoch timestamp
        
        self.routing_table.route_cache.store("A->Z", test_route)
        self.routing_table.cleanup_cache()
        
        self.assertNotIn("A->Z", self.routing_table.route_cache)

if __name__ == '__main__':
    unittest.main(failfast=True)
