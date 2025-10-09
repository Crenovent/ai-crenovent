"""
Task 6.3.24: Request dedup & coalescing
Implement request deduplication and coalescing for efficiency
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CoalescedRequest:
    """Coalesced request with multiple clients"""
    request_hash: str
    request_data: Dict[str, Any]
    client_futures: List[asyncio.Future]
    created_at: datetime
    timeout_seconds: int = 5

class RequestDedupService:
    """
    Request deduplication and coalescing service
    Task 6.3.24: Same key window efficiency
    """
    
    def __init__(self, coalescing_window_ms: int = 100):
        self.coalescing_window_ms = coalescing_window_ms
        self.pending_requests: Dict[str, CoalescedRequest] = {}
        self.processed_hashes: Dict[str, Any] = {}  # Cache for dedup
        self.cache_ttl_seconds = 60
    
    def _generate_request_hash(self, request_data: Dict[str, Any]) -> str:
        """Generate hash for request deduplication"""
        # Sort keys for consistent hashing
        normalized = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    async def process_request(self, request_data: Dict[str, Any]) -> Any:
        """Process request with deduplication and coalescing"""
        request_hash = self._generate_request_hash(request_data)
        
        # Check if we have a cached result (deduplication)
        if request_hash in self.processed_hashes:
            cache_entry = self.processed_hashes[request_hash]
            if datetime.utcnow() - cache_entry['timestamp'] < timedelta(seconds=self.cache_ttl_seconds):
                logger.info(f"Returning cached result for hash: {request_hash[:8]}")
                return cache_entry['result']
        
        # Check if request is already being processed (coalescing)
        if request_hash in self.pending_requests:
            logger.info(f"Coalescing request with hash: {request_hash[:8]}")
            future = asyncio.Future()
            self.pending_requests[request_hash].client_futures.append(future)
            return await future
        
        # Create new coalesced request
        future = asyncio.Future()
        coalesced_request = CoalescedRequest(
            request_hash=request_hash,
            request_data=request_data,
            client_futures=[future],
            created_at=datetime.utcnow()
        )
        
        self.pending_requests[request_hash] = coalesced_request
        
        # Wait for coalescing window
        await asyncio.sleep(self.coalescing_window_ms / 1000)
        
        # Process the coalesced request
        try:
            result = await self._execute_request(request_data)
            
            # Cache the result
            self.processed_hashes[request_hash] = {
                'result': result,
                'timestamp': datetime.utcnow()
            }
            
            # Notify all waiting clients
            for client_future in coalesced_request.client_futures:
                if not client_future.done():
                    client_future.set_result(result)
            
            return result
            
        except Exception as e:
            # Notify all waiting clients of the error
            for client_future in coalesced_request.client_futures:
                if not client_future.done():
                    client_future.set_exception(e)
            raise
        
        finally:
            # Clean up pending request
            if request_hash in self.pending_requests:
                del self.pending_requests[request_hash]
    
    async def _execute_request(self, request_data: Dict[str, Any]) -> Any:
        """Execute the actual request"""
        # Simulate request processing
        await asyncio.sleep(0.1)
        return {"result": "processed", "request_id": request_data.get("id", "unknown")}
    
    def cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, value in self.processed_hashes.items()
            if current_time - value['timestamp'] > timedelta(seconds=self.cache_ttl_seconds)
        ]
        
        for key in expired_keys:
            del self.processed_hashes[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# Global dedup service instance
request_dedup_service = RequestDedupService()
