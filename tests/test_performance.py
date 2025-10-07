"""
Performance Testing Framework for RevAI Pro Platform
Load testing, stress testing, and performance benchmarking
"""

import pytest
import asyncio
import httpx
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import json
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class PerformanceMetrics:
    """Performance metrics collection and analysis"""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.throughput_rates: List[float] = []
        self.error_rates: List[float] = []
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self.start_time: float = 0
        self.end_time: float = 0
    
    def start_test(self):
        """Start performance test"""
        self.start_time = time.time()
        self.response_times.clear()
        self.throughput_rates.clear()
        self.error_rates.clear()
        self.memory_usage.clear()
        self.cpu_usage.clear()
    
    def end_test(self):
        """End performance test"""
        self.end_time = time.time()
    
    def add_response_time(self, response_time: float):
        """Add response time measurement"""
        self.response_times.append(response_time)
    
    def add_throughput(self, requests_per_second: float):
        """Add throughput measurement"""
        self.throughput_rates.append(requests_per_second)
    
    def add_error_rate(self, error_rate: float):
        """Add error rate measurement"""
        self.error_rates.append(error_rate)
    
    def add_system_metrics(self):
        """Add current system metrics"""
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance test summary"""
        duration = self.end_time - self.start_time
        total_requests = len(self.response_times)
        
        summary = {
            "test_duration_seconds": duration,
            "total_requests": total_requests,
            "avg_response_time_ms": statistics.mean(self.response_times) if self.response_times else 0,
            "p50_response_time_ms": statistics.median(self.response_times) if self.response_times else 0,
            "p95_response_time_ms": self._percentile(self.response_times, 95),
            "p99_response_time_ms": self._percentile(self.response_times, 99),
            "max_response_time_ms": max(self.response_times) if self.response_times else 0,
            "min_response_time_ms": min(self.response_times) if self.response_times else 0,
            "avg_throughput_rps": statistics.mean(self.throughput_rates) if self.throughput_rates else 0,
            "max_throughput_rps": max(self.throughput_rates) if self.throughput_rates else 0,
            "avg_error_rate": statistics.mean(self.error_rates) if self.error_rates else 0,
            "max_error_rate": max(self.error_rates) if self.error_rates else 0,
            "avg_memory_usage_percent": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "max_memory_usage_percent": max(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_usage_percent": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu_usage_percent": max(self.cpu_usage) if self.cpu_usage else 0
        }
        
        return summary
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

class LoadTestRunner:
    """Load test runner for microservices"""
    
    def __init__(self, service_url: str, max_concurrent: int = 10):
        self.service_url = service_url
        self.max_concurrent = max_concurrent
        self.metrics = PerformanceMetrics()
    
    async def run_load_test(self, 
                          endpoint: str, 
                          payload: Dict[str, Any], 
                          duration_seconds: int = 60,
                          requests_per_second: int = 10) -> PerformanceMetrics:
        """Run load test against an endpoint"""
        
        self.metrics.start_test()
        
        # Calculate request interval
        request_interval = 1.0 / requests_per_second
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def make_request():
            async with semaphore:
                start_time = time.time()
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{self.service_url}{endpoint}",
                            json=payload,
                            timeout=30.0
                        )
                        response_time = (time.time() - start_time) * 1000  # Convert to ms
                        
                        self.metrics.add_response_time(response_time)
                        self.metrics.add_error_rate(0 if response.status_code == 200 else 1)
                        
                        return response.status_code == 200
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    self.metrics.add_response_time(response_time)
                    self.metrics.add_error_rate(1)
                    return False
        
        # Run load test
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Start request
            asyncio.create_task(make_request())
            request_count += 1
            
            # Add system metrics periodically
            if request_count % 10 == 0:
                self.metrics.add_system_metrics()
            
            # Wait for next request
            await asyncio.sleep(request_interval)
        
        # Wait for remaining requests to complete
        await asyncio.sleep(5)
        
        self.metrics.end_test()
        
        # Calculate throughput
        actual_duration = time.time() - start_time
        throughput = request_count / actual_duration
        self.metrics.add_throughput(throughput)
        
        return self.metrics

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.fixture
    def services(self):
        """Service endpoints"""
        return {
            "orchestrator": "http://localhost:8001",
            "agent_registry": "http://localhost:8002",
            "routing_orchestrator": "http://localhost:8003",
            "kpi_exporter": "http://localhost:8004",
            "confidence_thresholds": "http://localhost:8005",
            "model_audit": "http://localhost:8006",
            "calendar_automation": "http://localhost:8007",
            "letsmeet_automation": "http://localhost:8008",
            "cruxx_automation": "http://localhost:8009",
            "run_trace_schema": "http://localhost:8010",
            "dlq_replay_tooling": "http://localhost:8011",
            "metrics_exporter": "http://localhost:8012",
            "event_bus": "http://localhost:8013"
        }
    
    async def test_orchestrator_performance(self, services):
        """Test orchestrator performance under load"""
        runner = LoadTestRunner(services["orchestrator"], max_concurrent=20)
        
        payload = {
            "tenant_id": "perf-test-tenant",
            "user_id": "perf-test-user",
            "session_id": "perf-test-session",
            "service_name": "calendar",
            "operation_type": "create_event",
            "input_data": {"title": "Performance Test Event"},
            "context": {}
        }
        
        metrics = await runner.run_load_test(
            endpoint="/orchestrate",
            payload=payload,
            duration_seconds=30,
            requests_per_second=20
        )
        
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["avg_response_time_ms"] < 1000, f"Average response time too high: {summary['avg_response_time_ms']}ms"
        assert summary["p95_response_time_ms"] < 2000, f"P95 response time too high: {summary['p95_response_time_ms']}ms"
        assert summary["avg_error_rate"] < 0.01, f"Error rate too high: {summary['avg_error_rate']}"
        assert summary["avg_throughput_rps"] >= 15, f"Throughput too low: {summary['avg_throughput_rps']} RPS"
        
        print(f"Orchestrator Performance Summary: {json.dumps(summary, indent=2)}")
    
    async def test_event_bus_performance(self, services):
        """Test event bus performance under load"""
        runner = LoadTestRunner(services["event_bus"], max_concurrent=50)
        
        payload = {
            "event_id": "perf-test-event",
            "tenant_id": "perf-test-tenant",
            "event_type": "calendar.event.created",
            "topic": "calendar-events",
            "payload": {
                "event_id": "perf-test-event",
                "tenant_id": "perf-test-tenant",
                "user_id": "perf-test-user",
                "title": "Performance Test Event",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            },
            "headers": {},
            "metadata": {}
        }
        
        metrics = await runner.run_load_test(
            endpoint="/events/publish",
            payload=payload,
            duration_seconds=30,
            requests_per_second=50
        )
        
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["avg_response_time_ms"] < 500, f"Average response time too high: {summary['avg_response_time_ms']}ms"
        assert summary["p95_response_time_ms"] < 1000, f"P95 response time too high: {summary['p95_response_time_ms']}ms"
        assert summary["avg_error_rate"] < 0.01, f"Error rate too high: {summary['avg_error_rate']}"
        assert summary["avg_throughput_rps"] >= 40, f"Throughput too low: {summary['avg_throughput_rps']} RPS"
        
        print(f"Event Bus Performance Summary: {json.dumps(summary, indent=2)}")
    
    async def test_metrics_exporter_performance(self, services):
        """Test metrics exporter performance under load"""
        runner = LoadTestRunner(services["metrics_exporter"], max_concurrent=30)
        
        payload = {
            "tenant_id": "perf-test-tenant",
            "user_id": "perf-test-user",
            "session_id": "perf-test-session",
            "ui_mode": "agent",
            "service_name": "calendar",
            "operation_type": "create_event",
            "confidence_score": 0.9,
            "trust_score": 0.8
        }
        
        metrics = await runner.run_load_test(
            endpoint="/metrics/mode-adoption",
            payload=payload,
            duration_seconds=30,
            requests_per_second=30
        )
        
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["avg_response_time_ms"] < 200, f"Average response time too high: {summary['avg_response_time_ms']}ms"
        assert summary["p95_response_time_ms"] < 500, f"P95 response time too high: {summary['p95_response_time_ms']}ms"
        assert summary["avg_error_rate"] < 0.01, f"Error rate too high: {summary['avg_error_rate']}"
        assert summary["avg_throughput_rps"] >= 25, f"Throughput too low: {summary['avg_throughput_rps']} RPS"
        
        print(f"Metrics Exporter Performance Summary: {json.dumps(summary, indent=2)}")
    
    async def test_concurrent_service_performance(self, services):
        """Test performance with concurrent requests to multiple services"""
        
        async def test_service(service_name: str, service_url: str, endpoint: str, payload: Dict[str, Any]):
            """Test individual service"""
            runner = LoadTestRunner(service_url, max_concurrent=10)
            
            metrics = await runner.run_load_test(
                endpoint=endpoint,
                payload=payload,
                duration_seconds=20,
                requests_per_second=10
            )
            
            return service_name, metrics.get_summary()
        
        # Define test configurations
        test_configs = [
            ("agent_registry", services["agent_registry"], "/agents/query", {
                "tenant_id": "perf-test-tenant",
                "capabilities": ["calendar_management"],
                "status": "active"
            }),
            ("routing_orchestrator", services["routing_orchestrator"], "/routing/route", {
                "tenant_id": "perf-test-tenant",
                "user_id": "perf-test-user",
                "session_id": "perf-test-session",
                "service_name": "calendar",
                "operation_type": "create_event",
                "confidence_score": 0.8,
                "trust_score": 0.7,
                "context": {}
            }),
            ("confidence_thresholds", services["confidence_thresholds"], "/confidence/evaluate", {
                "tenant_id": "perf-test-tenant",
                "service_name": "calendar",
                "operation_type": "create_event",
                "confidence_score": 0.8,
                "trust_score": 0.7,
                "context": {}
            })
        ]
        
        # Run concurrent tests
        tasks = []
        for service_name, service_url, endpoint, payload in test_configs:
            task = test_service(service_name, service_url, endpoint, payload)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        for service_name, summary in results:
            print(f"{service_name} Performance Summary: {json.dumps(summary, indent=2)}")
            
            # Basic performance assertions
            assert summary["avg_response_time_ms"] < 1000, f"{service_name} response time too high"
            assert summary["avg_error_rate"] < 0.05, f"{service_name} error rate too high"
    
    async def test_memory_usage_under_load(self, services):
        """Test memory usage under sustained load"""
        
        # Get initial memory usage
        initial_memory = psutil.virtual_memory().percent
        
        # Run sustained load test
        runner = LoadTestRunner(services["orchestrator"], max_concurrent=50)
        
        payload = {
            "tenant_id": "memory-test-tenant",
            "user_id": "memory-test-user",
            "session_id": "memory-test-session",
            "service_name": "calendar",
            "operation_type": "create_event",
            "input_data": {"title": "Memory Test Event"},
            "context": {}
        }
        
        metrics = await runner.run_load_test(
            endpoint="/orchestrate",
            payload=payload,
            duration_seconds=60,
            requests_per_second=30
        )
        
        # Get final memory usage
        final_memory = psutil.virtual_memory().percent
        memory_increase = final_memory - initial_memory
        
        summary = metrics.get_summary()
        
        # Memory usage assertions
        assert memory_increase < 20, f"Memory usage increased too much: {memory_increase}%"
        assert summary["max_memory_usage_percent"] < 90, f"Maximum memory usage too high: {summary['max_memory_usage_percent']}%"
        
        print(f"Memory Usage Test Summary:")
        print(f"  Initial Memory: {initial_memory}%")
        print(f"  Final Memory: {final_memory}%")
        print(f"  Memory Increase: {memory_increase}%")
        print(f"  Max Memory Usage: {summary['max_memory_usage_percent']}%")
    
    async def test_cpu_usage_under_load(self, services):
        """Test CPU usage under sustained load"""
        
        # Get initial CPU usage
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Run sustained load test
        runner = LoadTestRunner(services["event_bus"], max_concurrent=100)
        
        payload = {
            "event_id": "cpu-test-event",
            "tenant_id": "cpu-test-tenant",
            "event_type": "calendar.event.created",
            "topic": "calendar-events",
            "payload": {
                "event_id": "cpu-test-event",
                "tenant_id": "cpu-test-tenant",
                "user_id": "cpu-test-user",
                "title": "CPU Test Event",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            },
            "headers": {},
            "metadata": {}
        }
        
        metrics = await runner.run_load_test(
            endpoint="/events/publish",
            payload=payload,
            duration_seconds=60,
            requests_per_second=50
        )
        
        # Get final CPU usage
        final_cpu = psutil.cpu_percent(interval=1)
        
        summary = metrics.get_summary()
        
        # CPU usage assertions
        assert summary["max_cpu_usage_percent"] < 95, f"Maximum CPU usage too high: {summary['max_cpu_usage_percent']}%"
        assert summary["avg_cpu_usage_percent"] < 80, f"Average CPU usage too high: {summary['avg_cpu_usage_percent']}%"
        
        print(f"CPU Usage Test Summary:")
        print(f"  Initial CPU: {initial_cpu}%")
        print(f"  Final CPU: {final_cpu}%")
        print(f"  Max CPU Usage: {summary['max_cpu_usage_percent']}%")
        print(f"  Avg CPU Usage: {summary['avg_cpu_usage_percent']}%")

class TestStressTesting:
    """Stress testing scenarios"""
    
    @pytest.fixture
    def services(self):
        """Service endpoints"""
        return {
            "orchestrator": "http://localhost:8001",
            "event_bus": "http://localhost:8013",
            "metrics_exporter": "http://localhost:8012"
        }
    
    async def test_gradual_load_increase(self, services):
        """Test gradual load increase to find breaking point"""
        
        # Test different load levels
        load_levels = [10, 25, 50, 75, 100, 150, 200]
        results = []
        
        for load_level in load_levels:
            print(f"Testing load level: {load_level} RPS")
            
            runner = LoadTestRunner(services["orchestrator"], max_concurrent=load_level)
            
            payload = {
                "tenant_id": "stress-test-tenant",
                "user_id": "stress-test-user",
                "session_id": "stress-test-session",
                "service_name": "calendar",
                "operation_type": "create_event",
                "input_data": {"title": f"Stress Test Event {load_level}"},
                "context": {}
            }
            
            metrics = await runner.run_load_test(
                endpoint="/orchestrate",
                payload=payload,
                duration_seconds=30,
                requests_per_second=load_level
            )
            
            summary = metrics.get_summary()
            results.append((load_level, summary))
            
            # Check if we've hit the breaking point
            if summary["avg_error_rate"] > 0.1:  # 10% error rate threshold
                print(f"Breaking point reached at {load_level} RPS")
                break
            
            # Wait between tests
            await asyncio.sleep(10)
        
        # Analyze results
        for load_level, summary in results:
            print(f"Load Level {load_level} RPS:")
            print(f"  Avg Response Time: {summary['avg_response_time_ms']:.2f}ms")
            print(f"  Error Rate: {summary['avg_error_rate']:.2%}")
            print(f"  Throughput: {summary['avg_throughput_rps']:.2f} RPS")
    
    async def test_burst_traffic_handling(self, services):
        """Test handling of burst traffic"""
        
        # Normal load
        normal_load = 20
        burst_load = 200
        
        print("Testing normal load...")
        runner = LoadTestRunner(services["event_bus"], max_concurrent=normal_load)
        
        payload = {
            "event_id": "burst-test-event",
            "tenant_id": "burst-test-tenant",
            "event_type": "calendar.event.created",
            "topic": "calendar-events",
            "payload": {
                "event_id": "burst-test-event",
                "tenant_id": "burst-test-tenant",
                "user_id": "burst-test-user",
                "title": "Burst Test Event",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            },
            "headers": {},
            "metadata": {}
        }
        
        # Run normal load for 30 seconds
        normal_metrics = await runner.run_load_test(
            endpoint="/events/publish",
            payload=payload,
            duration_seconds=30,
            requests_per_second=normal_load
        )
        
        print("Testing burst load...")
        
        # Run burst load for 10 seconds
        burst_runner = LoadTestRunner(services["event_bus"], max_concurrent=burst_load)
        burst_metrics = await burst_runner.run_load_test(
            endpoint="/events/publish",
            payload=payload,
            duration_seconds=10,
            requests_per_second=burst_load
        )
        
        # Analyze results
        normal_summary = normal_metrics.get_summary()
        burst_summary = burst_metrics.get_summary()
        
        print("Normal Load Summary:")
        print(f"  Avg Response Time: {normal_summary['avg_response_time_ms']:.2f}ms")
        print(f"  Error Rate: {normal_summary['avg_error_rate']:.2%}")
        print(f"  Throughput: {normal_summary['avg_throughput_rps']:.2f} RPS")
        
        print("Burst Load Summary:")
        print(f"  Avg Response Time: {burst_summary['avg_response_time_ms']:.2f}ms")
        print(f"  Error Rate: {burst_summary['avg_error_rate']:.2%}")
        print(f"  Throughput: {burst_summary['avg_throughput_rps']:.2f} RPS")
        
        # Assertions
        assert burst_summary["avg_error_rate"] < 0.2, "Burst load error rate too high"
        assert burst_summary["avg_response_time_ms"] < 5000, "Burst load response time too high"

class TestScalabilityTesting:
    """Scalability testing scenarios"""
    
    async def test_horizontal_scaling_simulation(self):
        """Simulate horizontal scaling by testing multiple service instances"""
        
        # This would typically involve:
        # 1. Starting multiple instances of services
        # 2. Load balancing requests across instances
        # 3. Measuring performance improvement
        
        # For now, we'll simulate by testing with different concurrency levels
        services = {
            "orchestrator": "http://localhost:8001",
            "event_bus": "http://localhost:8013"
        }
        
        concurrency_levels = [10, 20, 40, 80]
        results = []
        
        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")
            
            runner = LoadTestRunner(services["orchestrator"], max_concurrent=concurrency)
            
            payload = {
                "tenant_id": "scale-test-tenant",
                "user_id": "scale-test-user",
                "session_id": "scale-test-session",
                "service_name": "calendar",
                "operation_type": "create_event",
                "input_data": {"title": f"Scale Test Event {concurrency}"},
                "context": {}
            }
            
            metrics = await runner.run_load_test(
                endpoint="/orchestrate",
                payload=payload,
                duration_seconds=20,
                requests_per_second=concurrency
            )
            
            summary = metrics.get_summary()
            results.append((concurrency, summary))
            
            await asyncio.sleep(5)
        
        # Analyze scaling efficiency
        for concurrency, summary in results:
            efficiency = summary["avg_throughput_rps"] / concurrency
            print(f"Concurrency {concurrency}: {summary['avg_throughput_rps']:.2f} RPS (Efficiency: {efficiency:.2f})")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
