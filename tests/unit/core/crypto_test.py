import unittest
from parameterized import parameterized
from cryptography.exceptions import InvalidSignature
from core.crypto import DigitalSigner, KeyManager

class TestCryptographicOperations(unittest.TestCase):
    
    TEST_PAYLOADS = [
        b"simple_message",
        b"",
        b"x" * 1024 * 1024  # 1MB payload
    ]
    
    def setUp(self):
        self.key_mgr = KeyManager(key_rotation_interval=3600)
        self.signer = DigitalSigner(self.key_mgr)
        
    @parameterized.expand(TEST_PAYLOADS)
    def test_signing_verification_cycle(self, payload):
        signature = self.signer.sign(payload)
        self.assertTrue(self.signer.verify(payload, signature))
        
    def test_signature_tampering_detection(self):
        payload = b"critical_data"
        signature = self.signer.sign(payload)
        
        altered_payload = payload + b"tamper"
        with self.assertRaises(InvalidSignature):
            self.signer.verify(altered_payload, signature)
            
    @patch('core.crypto.time.time', return_value=0)
    def test_key_rotation_policy(self, mock_time):
        # Initial key
        key1 = self.key_mgr.current_key
        
        # Force key rotation
        mock_time.return_value = 3601
        key2 = self.key_mgr.current_key
        
        self.assertNotEqual(key1, key2)
        
    def test_concurrent_signing_operations(self):
        from concurrent.futures import ThreadPoolExecutor
        
        payload = b"stress_test"
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(self.signer.sign, payload) 
                      for _ in range(1000)]
            
            signatures = {f.result() for f in futures}
            
        # All signatures must be unique due to nonce usage
        self.assertEqual(len(signatures), 1000)

if __name__ == '__main__':
    unittest.main(verbosity=2)
