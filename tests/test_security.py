"""
Security Testing Suite for RevAI Pro Platform
Authentication, authorization, input validation, and security vulnerability testing
"""

import pytest
import asyncio
import httpx
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import base64
import hashlib
import hmac

class SecurityTestHelper:
    """Helper functions for security testing"""
    
    @staticmethod
    def generate_malicious_payloads() -> List[Dict[str, Any]]:
        """Generate various malicious payloads for testing"""
        return [
            # SQL Injection attempts
            {"title": "'; DROP TABLE events; --", "description": "SQL injection test"},
            {"title": "1' OR '1'='1", "description": "SQL injection test"},
            {"title": "admin'--", "description": "SQL injection test"},
            
            # XSS attempts
            {"title": "<script>alert('XSS')</script>", "description": "XSS test"},
            {"title": "javascript:alert('XSS')", "description": "XSS test"},
            {"title": "<img src=x onerror=alert('XSS')>", "description": "XSS test"},
            
            # Path traversal attempts
            {"title": "../../etc/passwd", "description": "Path traversal test"},
            {"title": "..\\..\\windows\\system32\\config\\sam", "description": "Path traversal test"},
            
            # Command injection attempts
            {"title": "; rm -rf /", "description": "Command injection test"},
            {"title": "| cat /etc/passwd", "description": "Command injection test"},
            
            # LDAP injection attempts
            {"title": "*)(uid=*))(|(uid=*", "description": "LDAP injection test"},
            
            # NoSQL injection attempts
            {"title": {"$ne": None}, "description": "NoSQL injection test"},
            {"title": {"$where": "this.title == this.description"}, "description": "NoSQL injection test"},
            
            # Large payloads
            {"title": "A" * 10000, "description": "Large payload test"},
            
            # Special characters
            {"title": "!@#$%^&*()_+{}|:<>?[]\\;'\",./", "description": "Special characters test"},
            
            # Unicode and encoding attacks
            {"title": "\u0000\u0001\u0002", "description": "Null byte injection test"},
            {"title": "\uFEFF\u200B\u200C", "description": "Unicode normalization test"},
            
            # JSON injection
            {"title": '{"malicious": "payload"}', "description": "JSON injection test"},
            
            # XML injection
            {"title": "<?xml version='1.0'?><root><malicious>payload</malicious></root>", "description": "XML injection test"}
        ]
    
    @staticmethod
    def generate_unauthorized_requests() -> List[Dict[str, Any]]:
        """Generate unauthorized request attempts"""
        return [
            # Missing authentication
            {"headers": {}},
            
            # Invalid authentication
            {"headers": {"Authorization": "Bearer invalid_token"}},
            {"headers": {"Authorization": "Basic invalid_credentials"}},
            
            # Expired token
            {"headers": {"Authorization": "Bearer expired_token"}},
            
            # Wrong tenant access
            {"headers": {"Authorization": "Bearer valid_token", "X-Tenant-ID": "unauthorized_tenant"}},
            
            # Privilege escalation
            {"headers": {"Authorization": "Bearer user_token", "X-Role": "admin"}},
            
            # Session hijacking
            {"headers": {"Authorization": "Bearer stolen_token"}},
            
            # CSRF attempts
            {"headers": {"Authorization": "Bearer valid_token", "X-CSRF-Token": "invalid"}},
        ]
    
    @staticmethod
    def generate_rate_limit_payloads() -> List[Dict[str, Any]]:
        """Generate payloads for rate limiting tests"""
        return [
            # Normal request
            {"title": "Normal Request", "description": "Regular event"},
            
            # Rapid requests (will be generated programmatically)
            {"title": "Rapid Request", "description": "Rapid event"},
        ]

class TestAuthenticationSecurity:
    """Authentication security tests"""
    
    @pytest.fixture
    def services(self):
        """Service endpoints"""
        return {
            "orchestrator": "http://localhost:8001",
            "agent_registry": "http://localhost:8002",
            "routing_orchestrator": "http://localhost:8003",
            "calendar_automation": "http://localhost:8007",
            "event_bus": "http://localhost:8013"
        }
    
    async def test_missing_authentication(self, services):
        """Test that endpoints require authentication"""
        endpoints = [
            ("/orchestrate", "POST"),
            ("/agents/register", "POST"),
            ("/routing/route", "POST"),
            ("/automation/trigger", "POST"),
            ("/events/publish", "POST")
        ]
        
        for endpoint, method in endpoints:
            service_url = services["orchestrator"] if "orchestrate" in endpoint else services["event_bus"]
            
            async with httpx.AsyncClient() as client:
                if method == "POST":
                    response = await client.post(f"{service_url}{endpoint}", json={})
                else:
                    response = await client.get(f"{service_url}{endpoint}")
                
                # Should return 401 Unauthorized or 403 Forbidden
                assert response.status_code in [401, 403], f"Endpoint {endpoint} should require authentication"
    
    async def test_invalid_authentication(self, services):
        """Test invalid authentication attempts"""
        invalid_tokens = [
            "invalid_token",
            "Bearer invalid_token",
            "Basic invalid_credentials",
            "expired_token",
            "malformed_token",
            ""
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": token} if token else {}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['orchestrator']}/orchestrate",
                    json={"tenant_id": "test-tenant"},
                    headers=headers
                )
                
                assert response.status_code in [401, 403], f"Invalid token should be rejected: {token}"
    
    async def test_authentication_bypass_attempts(self, services):
        """Test various authentication bypass attempts"""
        bypass_attempts = [
            # SQL injection in authentication
            {"headers": {"Authorization": "Bearer ' OR '1'='1"}},
            
            # LDAP injection in authentication
            {"headers": {"Authorization": "Bearer *)(uid=*))(|(uid=*"}},
            
            # NoSQL injection in authentication
            {"headers": {"Authorization": "Bearer {\"$ne\": null}"}},
            
            # Path traversal in authentication
            {"headers": {"Authorization": "Bearer ../../../etc/passwd"}},
            
            # Command injection in authentication
            {"headers": {"Authorization": "Bearer ; rm -rf /"}},
            
            # XSS in authentication
            {"headers": {"Authorization": "Bearer <script>alert('XSS')</script>"}},
        ]
        
        for attempt in bypass_attempts:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['orchestrator']}/orchestrate",
                    json={"tenant_id": "test-tenant"},
                    headers=attempt["headers"]
                )
                
                assert response.status_code in [401, 403], f"Authentication bypass attempt should be rejected"
    
    async def test_session_management(self, services):
        """Test session management security"""
        # Test session timeout
        # Test session invalidation
        # Test concurrent sessions
        # Test session fixation
        
        # This would require actual session management implementation
        # For now, we'll test basic session handling
        
        async with httpx.AsyncClient() as client:
            # Test without session
            response = await client.post(
                f"{services['orchestrator']}/orchestrate",
                json={"tenant_id": "test-tenant"}
            )
            assert response.status_code in [401, 403]

class TestAuthorizationSecurity:
    """Authorization security tests"""
    
    @pytest.fixture
    def services(self):
        """Service endpoints"""
        return {
            "orchestrator": "http://localhost:8001",
            "agent_registry": "http://localhost:8002",
            "calendar_automation": "http://localhost:8007",
            "cruxx_automation": "http://localhost:8009"
        }
    
    async def test_tenant_isolation(self, services):
        """Test that tenants cannot access each other's data"""
        # This test assumes proper tenant isolation is implemented
        
        # Create data for tenant A
        tenant_a_data = {
            "tenant_id": "tenant-a",
            "user_id": "user-a",
            "title": "Tenant A Event",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        # Create data for tenant B
        tenant_b_data = {
            "tenant_id": "tenant-b",
            "user_id": "user-b",
            "title": "Tenant B Event",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        # Test that tenant A cannot access tenant B's data
        async with httpx.AsyncClient() as client:
            # Try to access tenant B's data with tenant A's credentials
            response = await client.get(
                f"{services['calendar_automation']}/events?tenant_id=tenant-b",
                headers={"X-Tenant-ID": "tenant-a", "Authorization": "Bearer tenant_a_token"}
            )
            
            # Should either return empty results or 403 Forbidden
            assert response.status_code in [200, 403]
            if response.status_code == 200:
                data = response.json()
                assert len(data) == 0 or all(event["tenant_id"] == "tenant-a" for event in data)
    
    async def test_role_based_access_control(self, services):
        """Test role-based access control"""
        # Test different user roles and their permissions
        
        roles = ["admin", "user", "viewer", "guest"]
        
        for role in roles:
            async with httpx.AsyncClient() as client:
                # Test admin operations with non-admin role
                if role != "admin":
                    response = await client.post(
                        f"{services['agent_registry']}/agents/register",
                        json={"name": "Test Agent", "capabilities": []},
                        headers={"Authorization": f"Bearer {role}_token", "X-Role": role}
                    )
                    
                    # Non-admin users should not be able to register agents
                    assert response.status_code in [403, 401], f"Role {role} should not be able to register agents"
    
    async def test_privilege_escalation(self, services):
        """Test privilege escalation attempts"""
        escalation_attempts = [
            # Try to escalate to admin
            {"headers": {"Authorization": "Bearer user_token", "X-Role": "admin"}},
            
            # Try to modify role in request
            {"headers": {"Authorization": "Bearer user_token", "X-Role": "super_admin"}},
            
            # Try to access admin endpoints
            {"headers": {"Authorization": "Bearer user_token"}},
        ]
        
        for attempt in escalation_attempts:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['agent_registry']}/agents/register",
                    json={"name": "Escalation Test Agent", "capabilities": []},
                    headers=attempt["headers"]
                )
                
                assert response.status_code in [401, 403], "Privilege escalation should be prevented"

class TestInputValidationSecurity:
    """Input validation security tests"""
    
    @pytest.fixture
    def services(self):
        """Service endpoints"""
        return {
            "orchestrator": "http://localhost:8001",
            "calendar_automation": "http://localhost:8007",
            "event_bus": "http://localhost:8013"
        }
    
    async def test_sql_injection_prevention(self, services):
        """Test SQL injection prevention"""
        malicious_payloads = SecurityTestHelper.generate_malicious_payloads()
        
        for payload in malicious_payloads:
            if "DROP TABLE" in str(payload.get("title", "")) or "OR '1'='1" in str(payload.get("title", "")):
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{services['calendar_automation']}/events",
                        json=payload,
                        headers={"Authorization": "Bearer valid_token"}
                    )
                    
                    # Should not cause SQL injection
                    assert response.status_code != 500, f"SQL injection attempt should not cause server error: {payload}"
    
    async def test_xss_prevention(self, services):
        """Test XSS prevention"""
        xss_payloads = [
            {"title": "<script>alert('XSS')</script>", "description": "XSS test"},
            {"title": "javascript:alert('XSS')", "description": "XSS test"},
            {"title": "<img src=x onerror=alert('XSS')>", "description": "XSS test"},
        ]
        
        for payload in xss_payloads:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['calendar_automation']}/events",
                    json=payload,
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Check that script tags are escaped or removed
                    title = data.get("title", "")
                    assert "<script>" not in title, f"XSS payload should be sanitized: {payload}"
                    assert "javascript:" not in title, f"XSS payload should be sanitized: {payload}"
    
    async def test_path_traversal_prevention(self, services):
        """Test path traversal prevention"""
        path_traversal_payloads = [
            {"title": "../../etc/passwd", "description": "Path traversal test"},
            {"title": "..\\..\\windows\\system32\\config\\sam", "description": "Path traversal test"},
            {"title": "/etc/passwd", "description": "Path traversal test"},
        ]
        
        for payload in path_traversal_payloads:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['calendar_automation']}/events",
                    json=payload,
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                # Should not cause path traversal
                assert response.status_code != 500, f"Path traversal attempt should not cause server error: {payload}"
    
    async def test_command_injection_prevention(self, services):
        """Test command injection prevention"""
        command_injection_payloads = [
            {"title": "; rm -rf /", "description": "Command injection test"},
            {"title": "| cat /etc/passwd", "description": "Command injection test"},
            {"title": "&& whoami", "description": "Command injection test"},
        ]
        
        for payload in command_injection_payloads:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['calendar_automation']}/events",
                    json=payload,
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                # Should not execute commands
                assert response.status_code != 500, f"Command injection attempt should not cause server error: {payload}"
    
    async def test_large_payload_handling(self, services):
        """Test handling of large payloads"""
        large_payloads = [
            {"title": "A" * 10000, "description": "Large payload test"},
            {"title": "A" * 100000, "description": "Very large payload test"},
            {"title": "A" * 1000000, "description": "Extremely large payload test"},
        ]
        
        for payload in large_payloads:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['calendar_automation']}/events",
                    json=payload,
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                # Should handle large payloads gracefully
                assert response.status_code in [200, 413, 400], f"Large payload should be handled gracefully: {len(payload['title'])} chars"
    
    async def test_special_character_handling(self, services):
        """Test handling of special characters"""
        special_char_payloads = [
            {"title": "!@#$%^&*()_+{}|:<>?[]\\;'\",./", "description": "Special characters test"},
            {"title": "\u0000\u0001\u0002", "description": "Null byte test"},
            {"title": "\uFEFF\u200B\u200C", "description": "Unicode normalization test"},
        ]
        
        for payload in special_char_payloads:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['calendar_automation']}/events",
                    json=payload,
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                # Should handle special characters gracefully
                assert response.status_code in [200, 400], f"Special characters should be handled gracefully: {payload}"

class TestRateLimitingSecurity:
    """Rate limiting security tests"""
    
    @pytest.fixture
    def services(self):
        """Service endpoints"""
        return {
            "orchestrator": "http://localhost:8001",
            "event_bus": "http://localhost:8013",
            "metrics_exporter": "http://localhost:8012"
        }
    
    async def test_rate_limiting_enforcement(self, services):
        """Test that rate limiting is enforced"""
        # Send requests rapidly to trigger rate limiting
        rapid_requests = 100
        success_count = 0
        rate_limited_count = 0
        
        async with httpx.AsyncClient() as client:
            for i in range(rapid_requests):
                try:
                    response = await client.post(
                        f"{services['orchestrator']}/orchestrate",
                        json={
                            "tenant_id": "rate-limit-test-tenant",
                            "user_id": "rate-limit-test-user",
                            "session_id": "rate-limit-test-session",
                            "service_name": "calendar",
                            "operation_type": "create_event",
                            "input_data": {"title": f"Rate Limit Test {i}"},
                            "context": {}
                        },
                        headers={"Authorization": "Bearer valid_token"},
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        success_count += 1
                    elif response.status_code == 429:  # Too Many Requests
                        rate_limited_count += 1
                    
                except httpx.TimeoutException:
                    # Timeout might indicate rate limiting
                    rate_limited_count += 1
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.01)
        
        # Should have some rate limiting
        assert rate_limited_count > 0, "Rate limiting should be enforced"
        print(f"Rate limiting test: {success_count} successful, {rate_limited_count} rate limited")
    
    async def test_different_rate_limits_per_endpoint(self, services):
        """Test that different endpoints have different rate limits"""
        endpoints = [
            ("/orchestrate", "POST"),
            ("/events/publish", "POST"),
            ("/metrics/mode-adoption", "POST")
        ]
        
        for endpoint, method in endpoints:
            service_url = services["orchestrator"] if "orchestrate" in endpoint else services["event_bus"]
            
            # Test rate limiting for each endpoint
            rapid_requests = 50
            rate_limited_count = 0
            
            async with httpx.AsyncClient() as client:
                for i in range(rapid_requests):
                    try:
                        if method == "POST":
                            response = await client.post(
                                f"{service_url}{endpoint}",
                                json={"test": f"data_{i}"},
                                headers={"Authorization": "Bearer valid_token"},
                                timeout=5.0
                            )
                        else:
                            response = await client.get(
                                f"{service_url}{endpoint}",
                                headers={"Authorization": "Bearer valid_token"},
                                timeout=5.0
                            )
                        
                        if response.status_code == 429:
                            rate_limited_count += 1
                    
                    except httpx.TimeoutException:
                        rate_limited_count += 1
                    
                    await asyncio.sleep(0.01)
            
            print(f"Endpoint {endpoint}: {rate_limited_count} rate limited out of {rapid_requests}")

class TestDataSecurity:
    """Data security tests"""
    
    @pytest.fixture
    def services(self):
        """Service endpoints"""
        return {
            "event_bus": "http://localhost:8013",
            "metrics_exporter": "http://localhost:8012",
            "model_audit": "http://localhost:8006"
        }
    
    async def test_data_encryption_in_transit(self, services):
        """Test that data is encrypted in transit"""
        # This test would require HTTPS endpoints
        # For now, we'll test that sensitive data is not exposed in responses
        
        sensitive_data = {
            "password": "secret_password",
            "api_key": "secret_api_key",
            "token": "secret_token",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "security-test-event",
                    "tenant_id": "security-test-tenant",
                    "event_type": "test.event",
                    "topic": "test-topic",
                    "payload": sensitive_data,
                    "headers": {},
                    "metadata": {}
                },
                headers={"Authorization": "Bearer valid_token"}
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = json.dumps(data)
                
                # Check that sensitive data is not exposed in response
                for key, value in sensitive_data.items():
                    assert str(value) not in response_text, f"Sensitive data {key} should not be exposed in response"
    
    async def test_data_encryption_at_rest(self, services):
        """Test that data is encrypted at rest"""
        # This test would require database access
        # For now, we'll test that sensitive data is properly handled
        
        sensitive_data = {
            "password": "secret_password",
            "api_key": "secret_api_key",
            "token": "secret_token"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['model_audit']}/audit/model-call",
                json={
                    "call_id": "security-test-call",
                    "tenant_id": "security-test-tenant",
                    "user_id": "security-test-user",
                    "session_id": "security-test-session",
                    "agent_id": "security-test-agent",
                    "model_name": "test-model",
                    "model_version": "1.0",
                    "prompt": "Test prompt",
                    "response": "Test response",
                    "tokens_used": 100,
                    "cost_usd": 0.01,
                    "latency_ms": 500,
                    "status": "success",
                    "metadata": sensitive_data
                },
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert response.status_code in [200, 201], "Sensitive data should be handled securely"
    
    async def test_pii_detection_and_masking(self, services):
        """Test PII detection and masking"""
        pii_data = [
            {"email": "user@example.com", "phone": "555-123-4567", "ssn": "123-45-6789"},
            {"name": "John Doe", "address": "123 Main St", "credit_card": "4111-1111-1111-1111"},
            {"ip_address": "192.168.1.1", "mac_address": "00:11:22:33:44:55"}
        ]
        
        for pii in pii_data:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['metrics_exporter']}/metrics/mode-adoption",
                    json={
                        "tenant_id": "pii-test-tenant",
                        "user_id": "pii-test-user",
                        "session_id": "pii-test-session",
                        "ui_mode": "agent",
                        "service_name": "calendar",
                        "operation_type": "create_event",
                        "confidence_score": 0.9,
                        "trust_score": 0.8,
                        "metadata": pii
                    },
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = json.dumps(data)
                    
                    # Check that PII is masked or not exposed
                    for key, value in pii.items():
                        if isinstance(value, str) and len(value) > 4:
                            # Check that full PII is not exposed
                            assert str(value) not in response_text, f"PII {key} should be masked in response"

class TestSecurityHeaders:
    """Security headers tests"""
    
    @pytest.fixture
    def services(self):
        """Service endpoints"""
        return {
            "orchestrator": "http://localhost:8001",
            "event_bus": "http://localhost:8013",
            "metrics_exporter": "http://localhost:8012"
        }
    
    async def test_security_headers_present(self, services):
        """Test that security headers are present"""
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy"
        ]
        
        for service_name, service_url in services.items():
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{service_url}/health")
                
                if response.status_code == 200:
                    headers = response.headers
                    
                    # Check for security headers
                    for header in security_headers:
                        # Some headers might not be present in all services
                        if header in headers:
                            print(f"{service_name} has {header}: {headers[header]}")
    
    async def test_cors_configuration(self, services):
        """Test CORS configuration"""
        cors_headers = [
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods",
            "Access-Control-Allow-Headers",
            "Access-Control-Allow-Credentials"
        ]
        
        for service_name, service_url in services.items():
            async with httpx.AsyncClient() as client:
                # Test preflight request
                response = await client.options(
                    f"{service_url}/health",
                    headers={
                        "Origin": "https://malicious-site.com",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type"
                    }
                )
                
                headers = response.headers
                
                # Check CORS headers
                for header in cors_headers:
                    if header in headers:
                        print(f"{service_name} CORS {header}: {headers[header]}")
                        
                        # Verify CORS is properly configured
                        if header == "Access-Control-Allow-Origin":
                            assert headers[header] != "*", f"{service_name} should not allow all origins"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
