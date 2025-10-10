"""
Task 6.3.75: Change windows (no deploy during blackout)
Deployment scheduling service with blackout window management
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, time, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BlackoutWindow:
    """Deployment blackout window"""
    window_id: str
    name: str
    start_time: time
    end_time: time
    days_of_week: Set[int]  # 0=Monday, 6=Sunday
    timezone: str = "UTC"
    reason: str = ""
    enabled: bool = True

@dataclass
class DeploymentRequest:
    """Deployment request"""
    request_id: str
    model_id: str
    deployment_type: str  # "canary", "blue_green", "rolling"
    requested_at: datetime
    scheduled_at: Optional[datetime] = None
    status: str = "pending"  # pending, scheduled, blocked, completed

class ChangeWindowService:
    """
    Change window and deployment scheduling service
    Task 6.3.75: Stability with blackout enforcement
    """
    
    def __init__(self):
        self.blackout_windows: Dict[str, BlackoutWindow] = {}
        self.deployment_requests: Dict[str, DeploymentRequest] = {}
        self.deployment_queue: List[str] = []  # request_ids
        
        # Initialize default blackout windows
        self._initialize_default_blackouts()
    
    def _initialize_default_blackouts(self) -> None:
        """Initialize common blackout windows"""
        default_windows = [
            BlackoutWindow(
                window_id="business_hours",
                name="Business Hours Blackout",
                start_time=time(9, 0),  # 9 AM
                end_time=time(17, 0),   # 5 PM
                days_of_week={0, 1, 2, 3, 4},  # Monday-Friday
                reason="Avoid disruption during business hours"
            ),
            BlackoutWindow(
                window_id="weekend_maintenance",
                name="Weekend Maintenance Window",
                start_time=time(2, 0),   # 2 AM
                end_time=time(6, 0),     # 6 AM
                days_of_week={5, 6},     # Saturday-Sunday
                reason="Weekend maintenance activities",
                enabled=False  # Disabled by default
            ),
            BlackoutWindow(
                window_id="month_end",
                name="Month End Freeze",
                start_time=time(0, 0),   # All day
                end_time=time(23, 59),
                days_of_week={0, 1, 2, 3, 4, 5, 6},  # All days
                reason="Month-end financial processing",
                enabled=False  # Would be enabled programmatically
            )
        ]
        
        for window in default_windows:
            self.blackout_windows[window.window_id] = window
    
    def register_blackout_window(self, window: BlackoutWindow) -> bool:
        """Register a blackout window"""
        try:
            self.blackout_windows[window.window_id] = window
            logger.info(f"Registered blackout window: {window.window_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register blackout window: {e}")
            return False
    
    def submit_deployment_request(self, request: DeploymentRequest) -> bool:
        """Submit deployment request for scheduling"""
        try:
            # Check if immediate deployment is allowed
            if self._is_deployment_allowed(datetime.utcnow()):
                request.scheduled_at = datetime.utcnow()
                request.status = "scheduled"
                logger.info(f"Deployment {request.request_id} scheduled immediately")
            else:
                # Find next available window
                next_window = self._find_next_deployment_window()
                if next_window:
                    request.scheduled_at = next_window
                    request.status = "scheduled"
                    logger.info(f"Deployment {request.request_id} scheduled for {next_window}")
                else:
                    request.status = "blocked"
                    logger.warning(f"Deployment {request.request_id} blocked - no available windows")
            
            # Store request
            self.deployment_requests[request.request_id] = request
            
            # Add to queue if scheduled
            if request.status == "scheduled":
                self.deployment_queue.append(request.request_id)
                self.deployment_queue.sort(
                    key=lambda rid: self.deployment_requests[rid].scheduled_at or datetime.max
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit deployment request: {e}")
            return False
    
    def _is_deployment_allowed(self, deployment_time: datetime) -> bool:
        """Check if deployment is allowed at given time"""
        for window in self.blackout_windows.values():
            if not window.enabled:
                continue
            
            if self._is_time_in_blackout(deployment_time, window):
                logger.info(f"Deployment blocked by blackout window: {window.name}")
                return False
        
        return True
    
    def _is_time_in_blackout(self, deployment_time: datetime, window: BlackoutWindow) -> bool:
        """Check if time falls within blackout window"""
        # Check day of week (0=Monday)
        weekday = deployment_time.weekday()
        if weekday not in window.days_of_week:
            return False
        
        # Check time range
        current_time = deployment_time.time()
        
        if window.start_time <= window.end_time:
            # Same day window
            return window.start_time <= current_time <= window.end_time
        else:
            # Overnight window (crosses midnight)
            return current_time >= window.start_time or current_time <= window.end_time
    
    def _find_next_deployment_window(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Find next available deployment window"""
        if from_time is None:
            from_time = datetime.utcnow()
        
        # Look ahead up to 7 days
        for days_ahead in range(7):
            check_date = from_time + timedelta(days=days_ahead)
            
            # Check every hour of the day
            for hour in range(24):
                check_time = check_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                if check_time <= from_time:
                    continue  # Skip past times
                
                if self._is_deployment_allowed(check_time):
                    return check_time
        
        return None  # No window found in next 7 days
    
    def get_next_deployment_window(self) -> Optional[datetime]:
        """Get next available deployment window"""
        return self._find_next_deployment_window()
    
    def check_deployment_status(self, request_id: str) -> Optional[Dict]:
        """Check status of deployment request"""
        if request_id not in self.deployment_requests:
            return None
        
        request = self.deployment_requests[request_id]
        
        return {
            "request_id": request_id,
            "model_id": request.model_id,
            "deployment_type": request.deployment_type,
            "status": request.status,
            "requested_at": request.requested_at.isoformat(),
            "scheduled_at": request.scheduled_at.isoformat() if request.scheduled_at else None,
            "queue_position": self.deployment_queue.index(request_id) if request_id in self.deployment_queue else None
        }
    
    def force_deployment(self, request_id: str, override_reason: str) -> bool:
        """Force deployment outside of normal windows (emergency)"""
        if request_id not in self.deployment_requests:
            return False
        
        request = self.deployment_requests[request_id]
        request.scheduled_at = datetime.utcnow()
        request.status = "scheduled"
        
        # Log override
        logger.warning(
            f"EMERGENCY OVERRIDE: Deployment {request_id} forced outside blackout window. "
            f"Reason: {override_reason}"
        )
        
        # Move to front of queue
        if request_id in self.deployment_queue:
            self.deployment_queue.remove(request_id)
        self.deployment_queue.insert(0, request_id)
        
        return True
    
    def get_blackout_schedule(self, days_ahead: int = 7) -> List[Dict]:
        """Get blackout schedule for next N days"""
        schedule = []
        start_date = datetime.utcnow().date()
        
        for days in range(days_ahead):
            check_date = start_date + timedelta(days=days)
            day_blackouts = []
            
            for window in self.blackout_windows.values():
                if not window.enabled:
                    continue
                
                # Check if this window applies to this day
                weekday = check_date.weekday()
                if weekday in window.days_of_week:
                    day_blackouts.append({
                        "window_id": window.window_id,
                        "name": window.name,
                        "start_time": window.start_time.strftime("%H:%M"),
                        "end_time": window.end_time.strftime("%H:%M"),
                        "reason": window.reason
                    })
            
            schedule.append({
                "date": check_date.isoformat(),
                "weekday": check_date.strftime("%A"),
                "blackout_windows": day_blackouts
            })
        
        return schedule
    
    def get_deployment_queue_status(self) -> Dict:
        """Get current deployment queue status"""
        queue_details = []
        
        for request_id in self.deployment_queue:
            request = self.deployment_requests[request_id]
            queue_details.append({
                "request_id": request_id,
                "model_id": request.model_id,
                "deployment_type": request.deployment_type,
                "scheduled_at": request.scheduled_at.isoformat() if request.scheduled_at else None,
                "status": request.status
            })
        
        return {
            "queue_length": len(self.deployment_queue),
            "next_deployment": queue_details[0] if queue_details else None,
            "queue_details": queue_details,
            "active_blackouts": len([w for w in self.blackout_windows.values() if w.enabled])
        }
    
    def enable_blackout_window(self, window_id: str) -> bool:
        """Enable a blackout window"""
        if window_id not in self.blackout_windows:
            return False
        
        self.blackout_windows[window_id].enabled = True
        logger.info(f"Enabled blackout window: {window_id}")
        return True
    
    def disable_blackout_window(self, window_id: str) -> bool:
        """Disable a blackout window"""
        if window_id not in self.blackout_windows:
            return False
        
        self.blackout_windows[window_id].enabled = False
        logger.info(f"Disabled blackout window: {window_id}")
        return True

# Global change window service
change_window_service = ChangeWindowService()
