"""
Task 6.3.61: Smoke probes (periodic health inference)
Add smoke probes for early outage detection with non-PII payloads
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ProbeStatus(Enum):
    """Smoke probe status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    ERROR = "error"

@dataclass
class SmokeProbeConfig:
    """Smoke probe configuration"""
    probe_id: str
    model_id: str
    interval_seconds: int
    timeout_seconds: int
    test_payload: Dict[str, Any]
    expected_response_fields: List[str]
    health_thresholds: Dict[str, float]
    enabled: bool = True

@dataclass
class ProbeResult:
    """Individual probe execution result"""
    probe_id: str
    model_id: str
    timestamp: datetime
    status: ProbeStatus
    response_time_ms: int
    success: bool
    error_message: Optional[str]
    response_data: Optional[Dict[str, Any]]

class SmokeProbeService:
    """
    Smoke probe service for early outage detection
    Task 6.3.61: Non-PII periodic health checks
    """
    
    def __init__(self):
        self.probe_configs: Dict[str, SmokeProbeConfig] = {}
        self.probe_results: Dict[str, List[ProbeResult]] = {}
        self.running_probes: Dict[str, asyncio.Task] = {}
        self.result_retention_hours = 24
    
    def register_smoke_probe(self, config: SmokeProbeConfig) -> bool:
        """Register a smoke probe configuration"""
        try:
            # Validate test payload is non-PII
            if not self._validate_non_pii_payload(config.test_payload):
                raise ValueError("Test payload contains potential PII data")
            
            self.probe_configs[config.probe_id] = config
            self.probe_results[config.probe_id] = []
            
            logger.info(f"Registered smoke probe: {config.probe_id} for model: {config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register smoke probe {config.probe_id}: {e}")
            return False
    
    async def start_probe(self, probe_id: str) -> bool:
        """Start a smoke probe"""
        if probe_id not in self.probe_configs:
            logger.error(f"Probe configuration not found: {probe_id}")
            return False
        
        if probe_id in self.running_probes:
            logger.warning(f"Probe {probe_id} is already running")
            return True
        
        config = self.probe_configs[probe_id]
        if not config.enabled:
            logger.info(f"Probe {probe_id} is disabled")
            return False
        
        # Start probe task
        probe_task = asyncio.create_task(self._run_probe_loop(config))
        self.running_probes[probe_id] = probe_task
        
        logger.info(f"Started smoke probe: {probe_id}")
        return True
    
    async def stop_probe(self, probe_id: str) -> bool:
        """Stop a smoke probe"""
        if probe_id not in self.running_probes:
            logger.warning(f"Probe {probe_id} is not running")
            return True
        
        probe_task = self.running_probes[probe_id]
        probe_task.cancel()
        
        try:
            await probe_task
        except asyncio.CancelledError:
            pass
        
        del self.running_probes[probe_id]
        logger.info(f"Stopped smoke probe: {probe_id}")
        return True
    
    async def _run_probe_loop(self, config: SmokeProbeConfig) -> None:
        """Run probe loop for continuous monitoring"""
        while True:
            try:
                # Execute probe
                result = await self._execute_probe(config)
                
                # Store result
                self.probe_results[config.probe_id].append(result)
                
                # Cleanup old results
                self._cleanup_old_results(config.probe_id)
                
                # Log status changes
                self._log_status_change(result)
                
                # Wait for next interval
                await asyncio.sleep(config.interval_seconds)
                
            except asyncio.CancelledError:
                logger.info(f"Probe loop cancelled for: {config.probe_id}")
                break
            except Exception as e:
                logger.error(f"Probe loop error for {config.probe_id}: {e}")
                await asyncio.sleep(config.interval_seconds)
    
    async def _execute_probe(self, config: SmokeProbeConfig) -> ProbeResult:
        """Execute a single probe"""
        start_time = datetime.utcnow()
        
        try:
            # Execute model inference with timeout
            response_data = await asyncio.wait_for(
                self._invoke_model(config.model_id, config.test_payload),
                timeout=config.timeout_seconds
            )
            
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Validate response
            validation_result = self._validate_response(response_data, config)
            
            return ProbeResult(
                probe_id=config.probe_id,
                model_id=config.model_id,
                timestamp=start_time,
                status=validation_result["status"],
                response_time_ms=response_time_ms,
                success=validation_result["success"],
                error_message=validation_result.get("error"),
                response_data=response_data
            )
            
        except asyncio.TimeoutError:
            response_time_ms = config.timeout_seconds * 1000
            return ProbeResult(
                probe_id=config.probe_id,
                model_id=config.model_id,
                timestamp=start_time,
                status=ProbeStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                success=False,
                error_message=f"Timeout after {config.timeout_seconds}s",
                response_data=None
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return ProbeResult(
                probe_id=config.probe_id,
                model_id=config.model_id,
                timestamp=start_time,
                status=ProbeStatus.ERROR,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e),
                response_data=None
            )
    
    async def _invoke_model(self, model_id: str, test_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke model for smoke test"""
        # Simulate model inference
        await asyncio.sleep(0.05)  # Simulate processing time
        
        return {
            "prediction": 0.75,
            "confidence": 0.85,
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
    
    def _validate_response(self, response_data: Dict[str, Any], config: SmokeProbeConfig) -> Dict[str, Any]:
        """Validate probe response against configuration"""
        if not response_data:
            return {
                "success": False,
                "status": ProbeStatus.ERROR,
                "error": "Empty response"
            }
        
        # Check required fields
        missing_fields = []
        for field in config.expected_response_fields:
            if field not in response_data:
                missing_fields.append(field)
        
        if missing_fields:
            return {
                "success": False,
                "status": ProbeStatus.UNHEALTHY,
                "error": f"Missing required fields: {missing_fields}"
            }
        
        # Check health thresholds
        for field, threshold in config.health_thresholds.items():
            if field in response_data:
                value = response_data[field]
                if isinstance(value, (int, float)) and value < threshold:
                    return {
                        "success": False,
                        "status": ProbeStatus.DEGRADED,
                        "error": f"Field {field} below threshold: {value} < {threshold}"
                    }
        
        return {
            "success": True,
            "status": ProbeStatus.HEALTHY
        }
    
    def _validate_non_pii_payload(self, payload: Dict[str, Any]) -> bool:
        """Validate that test payload contains no PII"""
        # Simple PII detection - check for common PII field names
        pii_fields = {
            'email', 'phone', 'ssn', 'social_security', 'credit_card',
            'name', 'first_name', 'last_name', 'address', 'zip_code',
            'postal_code', 'ip_address', 'user_id', 'customer_id'
        }
        
        def check_dict(d: Dict[str, Any]) -> bool:
            for key, value in d.items():
                # Check key names
                if key.lower() in pii_fields:
                    return False
                
                # Check nested dictionaries
                if isinstance(value, dict):
                    if not check_dict(value):
                        return False
                
                # Check string values for patterns
                if isinstance(value, str):
                    if self._contains_pii_pattern(value):
                        return False
            
            return True
        
        return check_dict(payload)
    
    def _contains_pii_pattern(self, value: str) -> bool:
        """Check if string contains PII patterns"""
        import re
        
        # Email pattern
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', value):
            return True
        
        # Phone pattern
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', value):
            return True
        
        # SSN pattern
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', value):
            return True
        
        return False
    
    def _cleanup_old_results(self, probe_id: str) -> None:
        """Clean up old probe results"""
        if probe_id not in self.probe_results:
            return
        
        cutoff_time = datetime.utcnow() - timedelta(hours=self.result_retention_hours)
        
        self.probe_results[probe_id] = [
            result for result in self.probe_results[probe_id]
            if result.timestamp > cutoff_time
        ]
    
    def _log_status_change(self, result: ProbeResult) -> None:
        """Log significant status changes"""
        if not result.success:
            logger.warning(
                f"Smoke probe {result.probe_id} failed: {result.error_message} "
                f"(response time: {result.response_time_ms}ms)"
            )
        elif result.status == ProbeStatus.DEGRADED:
            logger.warning(f"Smoke probe {result.probe_id} showing degraded performance")
    
    def get_probe_status(self, probe_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a probe"""
        if probe_id not in self.probe_results or not self.probe_results[probe_id]:
            return None
        
        recent_results = self.probe_results[probe_id][-10:]  # Last 10 results
        latest_result = recent_results[-1]
        
        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
        avg_response_time = sum(r.response_time_ms for r in recent_results) / len(recent_results)
        
        return {
            "probe_id": probe_id,
            "model_id": latest_result.model_id,
            "current_status": latest_result.status.value,
            "last_check": latest_result.timestamp.isoformat(),
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time,
            "is_running": probe_id in self.running_probes,
            "recent_error": latest_result.error_message if not latest_result.success else None
        }
    
    async def start_all_probes(self) -> Dict[str, bool]:
        """Start all registered probes"""
        results = {}
        for probe_id in self.probe_configs:
            results[probe_id] = await self.start_probe(probe_id)
        return results
    
    async def stop_all_probes(self) -> Dict[str, bool]:
        """Stop all running probes"""
        results = {}
        for probe_id in list(self.running_probes.keys()):
            results[probe_id] = await self.stop_probe(probe_id)
        return results

# Global smoke probe service instance
smoke_probe_service = SmokeProbeService()
