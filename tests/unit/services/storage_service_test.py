import unittest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
from services.storage_service import DistributedKVStore, ACIDValidator

class TestStorageService(unittest.TestCase):
    
    def setUp(self):
        self.backend = MagicMock()
        self.kv_store = DistributedKVStore(
            backend=self.backend,
            replication_factor=3
        )
        self.validator = ACIDValidator()
        
    def test_crud_operations(self):
        # Create
        self.kv_store.put("test_key", "enterprise_data")
        self.backend.write.assert_called_with("test_key", "enterprise_data", replicas=3)
        
        # Read
        self.backend.read.return_value = "enterprise_data"
        self.assertEqual(self.kv_store.get("test_key"), "enterprise_data")
        
        # Delete
        self.kv_store.delete("test_key")
        self.backend.delete.assert_called_with("test_key", quorum=2)
        
    def test_transaction_rollback(self):
        @contextmanager
        def fake_transaction():
            yield
            raise Exception("Simulated failure")
            
        self.backend.start_transaction.side_effect = fake_transaction
        
        with self.assertRaises(Exception):
            with self.kv_store.transaction():
                self.kv_store.put("temp", "data")
                
        self.backend.rollback.assert_called_once()
        
    @patch('services.storage_service.ConsistencyChecker.verify_quorum')
    def test_consistency_validation(self, mock_verify):
        mock_verify.return_value = True
        test_data = [
            ("node1", "v1"),
            ("node2", "v1"),
            ("node3", "v1")
        ]
        self.assertTrue(self.validator.check_consistency(test_data))
        
        test_data[2] = ("node3", "v2")
        self.assertFalse(self.validator.check_consistency(test_data))

if __name__ == '__main__':
    unittest.main(verbosity=2)
