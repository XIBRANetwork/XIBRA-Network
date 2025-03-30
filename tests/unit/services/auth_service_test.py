import unittest
from datetime import datetime, timedelta
from parameterized import parameterized
from jose import jwt, JWTError
from services.auth_service import JWTManager, RBACValidator

class TestAuthService(unittest.TestCase):
    
    TEST_TOKENS = [
        ({"sub": "admin"}, "HS256", 3600, "valid_admin"),
        ({"sub": "user", "scopes": ["read"]}, "RS256", 0, "expired_user"),
        (None, "HS256", 3600, "invalid_payload")
    ]
    
    def setUp(self):
        self.jwt_mgr = JWTManager(
            secret_key="xibra_enterprise_key",
            algorithm="HS256"
        )
        self.rbac = RBACValidator(policy_file="xibra_policies.json")
        
    @parameterized.expand(TEST_TOKENS)
    def test_jwt_generation_validation(self, payload, algo, exp, case):
        if payload:
            token = self.jwt_mgr.create_token(
                payload, 
                expires_delta=timedelta(seconds=exp)
            )
            decoded = self.jwt_mgr.verify_token(token)
            
            if exp > 0:
                self.assertEqual(decoded["sub"], payload["sub"])
            else:
                with self.assertRaises(jwt.ExpiredSignatureError):
                    self.jwt_mgr.verify_token(token)
        else:
            with self.assertRaises(ValueError):
                self.jwt_mgr.create_token(payload)
                
    def test_role_based_access_control(self):
        test_cases = [
            ("admin", "cluster/delete", True),
            ("user", "dataset/create", False),
            ("auditor", "logs/view", True)
        ]
        
        for role, resource, expected in test_cases:
            with self.subTest(f"{role}_accessing_{resource}"):
                result = self.rbac.check_permission(role, resource)
                self.assertEqual(result, expected)
                
    @patch('services.auth_service.requests.get')
    def test_oidc_integration(self, mock_get):
        mock_get.return_value.json.return_value = {
            "jwks_uri": "https://auth.xibra.net/certs"
        }
        
        oidc_token = MagicMock()
        decoded = self.jwt_mgr.verify_oidc_token(oidc_token)
        mock_get.assert_called_with("https://auth.xibra.net/.well-known/openid-configuration")

if __name__ == '__main__':
    unittest.main(failfast=True)
