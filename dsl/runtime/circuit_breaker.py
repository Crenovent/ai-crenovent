# Circuit Breaker Pattern for RBA Runtime
# Task 6.2-T17: Implement circuit breaker for repeated errors

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker tripped, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: int = 60          # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Operation timeout in seconds
    expected_exception: type = Exception # Exception type to monitor

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_requests: int = 0
    failed_requests: int = 0
    successful_requests: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changed_at: float = 0

class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """
    Circuit breaker implementation for protecting RBA runtime from cascading failures
    """
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.stats.state_changed_at = time.time()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        """
        async with self._lock:
            current_time = time.time()
            
            # Check if we should transition from OPEN to HALF_OPEN
            if (self.state == CircuitState.OPEN and 
                current_time - self.stats.state_changed_at >= self.config.recovery_timeout):
                self._transition_to_half_open()
            
            # Fail fast if circuit is open
            if self.state == CircuitState.OPEN:
                self.stats.total_requests += 1
                self.stats.failed_requests += 1
                logger.warning(f"Circuit breaker {self.name} is OPEN - failing fast")
                raise CircuitBreakerError(f"Circuit breaker {self.name} is open")
        
        # Execute the function
        try:
            # Apply timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            await self._record_success()
            return result
            
        except asyncio.TimeoutError as e:
            await self._record_failure()
            raise CircuitBreakerError(f"Operation timed out after {self.config.timeout}s") from e
            
        except self.config.expected_exception as e:
            await self._record_failure()
            raise e
        
        except Exception as e:
            # Unexpected exception - don't count towards circuit breaker
            logger.error(f"Unexpected exception in circuit breaker {self.name}: {e}")
            raise e
    
    async def _record_success(self) -> None:
        """Record successful operation"""
        async with self._lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes += 1
            
            # Transition from HALF_OPEN to CLOSED if enough successes
            if (self.state == CircuitState.HALF_OPEN and 
                self.stats.consecutive_successes >= self.config.success_threshold):
                self._transition_to_closed()
                logger.info(f"Circuit breaker {self.name} transitioned to CLOSED after {self.stats.consecutive_successes} successes")
    
    async def _record_failure(self) -> None:
        """Record failed operation"""
        async with self._lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.last_failure_time = time.time()
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            
            # Transition to OPEN if failure threshold exceeded
            if (self.state == CircuitState.CLOSED and 
                self.stats.consecutive_failures >= self.config.failure_threshold):
                self._transition_to_open()
                logger.error(f"Circuit breaker {self.name} transitioned to OPEN after {self.stats.consecutive_failures} failures")
            
            # Transition from HALF_OPEN back to OPEN on any failure
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
                logger.warning(f"Circuit breaker {self.name} transitioned back to OPEN from HALF_OPEN")
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state"""
        self.state = CircuitState.OPEN
        self.stats.state_changed_at = time.time()
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.stats.state_changed_at = time.time()
        self.stats.consecutive_successes = 0
        logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN for testing")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.stats.state_changed_at = time.time()
        self.stats.consecutive_failures = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'total_requests': self.stats.total_requests,
            'successful_requests': self.stats.successful_requests,
            'failed_requests': self.stats.failed_requests,
            'failure_rate': (
                self.stats.failed_requests / self.stats.total_requests 
                if self.stats.total_requests > 0 else 0
            ),
            'consecutive_failures': self.stats.consecutive_failures,
            'consecutive_successes': self.stats.consecutive_successes,
            'last_failure_time': self.stats.last_failure_time,
            'state_changed_at': self.stats.state_changed_at,
            'uptime_seconds': time.time() - self.stats.state_changed_at
        }

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_configs: Dict[str, CircuitBreakerConfig] = {
            'database': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2,
                timeout=10.0
            ),
            'external_api': CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60,
                success_threshold=3,
                timeout=30.0
            ),
            'ml_model': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=120,
                success_threshold=2,
                timeout=60.0
            ),
            'notification': CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout=30,
                success_threshold=5,
                timeout=15.0
            )
        }
    
    def get_circuit_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            if config is None:
                config = self.default_configs.get(name, CircuitBreakerConfig())
            
            self.circuit_breakers[name] = CircuitBreaker(config, name)
            logger.info(f"Created circuit breaker: {name}")
        
        return self.circuit_breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: cb.get_stats() 
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_circuit_breaker(self, name: str) -> bool:
        """Reset circuit breaker to CLOSED state"""
        if name in self.circuit_breakers:
            cb = self.circuit_breakers[name]
            cb._transition_to_closed()
            cb.stats = CircuitBreakerStats()
            cb.stats.state_changed_at = time.time()
            logger.info(f"Reset circuit breaker: {name}")
            return True
        return False

# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()

# Decorator for easy circuit breaker usage
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator to add circuit breaker protection to functions
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cb = circuit_breaker_registry.get_circuit_breaker(name, config)
            return await cb.call(func, *args, **kwargs)
        return wrapper
    return decorator

# Context manager for circuit breaker
class circuit_breaker_context:
    """
    Context manager for circuit breaker protection
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config
        self.circuit_breaker = None
    
    async def __aenter__(self):
        self.circuit_breaker = circuit_breaker_registry.get_circuit_breaker(
            self.name, self.config
        )
        return self.circuit_breaker
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.circuit_breaker:
            await self.circuit_breaker._record_failure()
        elif self.circuit_breaker:
            await self.circuit_breaker._record_success()

# Example usage patterns
class RBACircuitBreakerIntegration:
    """
    Integration of circuit breakers into RBA runtime components
    """
    
    @staticmethod
    async def execute_with_database_protection(operation: Callable) -> Any:
        """Execute database operation with circuit breaker protection"""
        cb = circuit_breaker_registry.get_circuit_breaker('database')
        return await cb.call(operation)
    
    @staticmethod
    async def execute_with_api_protection(operation: Callable) -> Any:
        """Execute external API call with circuit breaker protection"""
        cb = circuit_breaker_registry.get_circuit_breaker('external_api')
        return await cb.call(operation)
    
    @staticmethod
    async def execute_with_ml_protection(operation: Callable) -> Any:
        """Execute ML model call with circuit breaker protection"""
        cb = circuit_breaker_registry.get_circuit_breaker('ml_model')
        return await cb.call(operation)
    
    @staticmethod
    async def execute_with_notification_protection(operation: Callable) -> Any:
        """Execute notification with circuit breaker protection"""
        cb = circuit_breaker_registry.get_circuit_breaker('notification')
        return await cb.call(operation)

# Health check endpoint for circuit breakers
async def get_circuit_breaker_health() -> Dict[str, Any]:
    """
    Get health status of all circuit breakers
    """
    all_stats = circuit_breaker_registry.get_all_stats()
    
    overall_health = "healthy"
    open_circuits = []
    
    for name, stats in all_stats.items():
        if stats['state'] == 'open':
            overall_health = "degraded"
            open_circuits.append(name)
        elif stats['state'] == 'half_open':
            if overall_health == "healthy":
                overall_health = "recovering"
    
    return {
        'overall_health': overall_health,
        'open_circuits': open_circuits,
        'circuit_breakers': all_stats,
        'timestamp': datetime.now().isoformat()
    }
