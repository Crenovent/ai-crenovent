"""
Task 6.3.63: Preemption policy for burst GPU jobs
Priority-based preemption system for GPU resource management
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class JobPriority(Enum):
    """Job priority levels"""
    CRITICAL = "critical"      # P0 - Never preempted
    HIGH = "high"             # P1 - Preempts lower priority
    NORMAL = "normal"         # P2 - Standard priority
    LOW = "low"               # P3 - Can be preempted
    BATCH = "batch"           # P4 - Lowest priority

@dataclass
class GPUJob:
    """GPU job configuration"""
    job_id: str
    model_id: str
    priority: JobPriority
    gpu_memory_required: int  # MB
    estimated_duration: int   # seconds
    tenant_id: str
    created_at: datetime
    started_at: Optional[datetime] = None
    preempted_count: int = 0

class PreemptionPolicyService:
    """
    GPU job preemption policy service
    Task 6.3.63: SLO compliance with priority-based eviction
    """
    
    def __init__(self):
        self.priority_weights = {
            JobPriority.CRITICAL: 1000,
            JobPriority.HIGH: 100,
            JobPriority.NORMAL: 10,
            JobPriority.LOW: 1,
            JobPriority.BATCH: 0
        }
        
        self.running_jobs: Dict[str, GPUJob] = {}
        self.queued_jobs: List[GPUJob] = []
        self.preempted_jobs: List[GPUJob] = []
        
        # GPU resource tracking
        self.total_gpu_memory = 24000  # 24GB per GPU
        self.available_gpu_memory = 24000
        self.gpu_count = 4
    
    def submit_job(self, job: GPUJob) -> bool:
        """Submit job for GPU processing"""
        try:
            # Check if job can run immediately
            if self._can_allocate_resources(job):
                return self._start_job(job)
            
            # Check if we can preempt lower priority jobs
            preemptable_jobs = self._find_preemptable_jobs(job)
            
            if preemptable_jobs:
                # Preempt lower priority jobs
                for preempted_job in preemptable_jobs:
                    self._preempt_job(preempted_job)
                
                # Start the new job
                return self._start_job(job)
            
            # Queue the job
            self._queue_job(job)
            logger.info(f"Queued job {job.job_id} due to resource constraints")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit job {job.job_id}: {e}")
            return False
    
    def _can_allocate_resources(self, job: GPUJob) -> bool:
        """Check if resources are available for job"""
        return self.available_gpu_memory >= job.gpu_memory_required
    
    def _find_preemptable_jobs(self, new_job: GPUJob) -> List[GPUJob]:
        """Find jobs that can be preempted for the new job"""
        if new_job.priority in [JobPriority.LOW, JobPriority.BATCH]:
            # Low priority jobs cannot preempt others
            return []
        
        preemptable = []
        memory_needed = new_job.gpu_memory_required - self.available_gpu_memory
        
        if memory_needed <= 0:
            return []
        
        # Sort running jobs by preemption priority (lowest first)
        candidates = sorted(
            self.running_jobs.values(),
            key=lambda j: (
                self.priority_weights[j.priority],
                j.started_at or datetime.utcnow()
            )
        )
        
        memory_freed = 0
        for job in candidates:
            # Can only preempt lower priority jobs
            if self.priority_weights[job.priority] < self.priority_weights[new_job.priority]:
                preemptable.append(job)
                memory_freed += job.gpu_memory_required
                
                if memory_freed >= memory_needed:
                    break
        
        return preemptable if memory_freed >= memory_needed else []
    
    def _start_job(self, job: GPUJob) -> bool:
        """Start running a job"""
        try:
            if not self._can_allocate_resources(job):
                return False
            
            # Allocate resources
            self.available_gpu_memory -= job.gpu_memory_required
            job.started_at = datetime.utcnow()
            
            # Move to running jobs
            self.running_jobs[job.job_id] = job
            
            logger.info(f"Started job {job.job_id} with priority {job.priority.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start job {job.job_id}: {e}")
            return False
    
    def _preempt_job(self, job: GPUJob) -> bool:
        """Preempt a running job"""
        try:
            if job.job_id not in self.running_jobs:
                return False
            
            # Free resources
            self.available_gpu_memory += job.gpu_memory_required
            
            # Update job state
            job.preempted_count += 1
            job.started_at = None
            
            # Move from running to preempted
            del self.running_jobs[job.job_id]
            self.preempted_jobs.append(job)
            
            logger.info(f"Preempted job {job.job_id} (count: {job.preempted_count})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to preempt job {job.job_id}: {e}")
            return False
    
    def _queue_job(self, job: GPUJob) -> None:
        """Add job to queue in priority order"""
        # Insert job in priority order
        inserted = False
        for i, queued_job in enumerate(self.queued_jobs):
            if self.priority_weights[job.priority] > self.priority_weights[queued_job.priority]:
                self.queued_jobs.insert(i, job)
                inserted = True
                break
        
        if not inserted:
            self.queued_jobs.append(job)
    
    def complete_job(self, job_id: str) -> bool:
        """Mark job as completed and free resources"""
        if job_id not in self.running_jobs:
            return False
        
        job = self.running_jobs[job_id]
        
        # Free resources
        self.available_gpu_memory += job.gpu_memory_required
        
        # Remove from running jobs
        del self.running_jobs[job_id]
        
        # Try to start queued jobs
        self._process_queue()
        
        logger.info(f"Completed job {job_id}")
        return True
    
    def _process_queue(self) -> None:
        """Process queued jobs"""
        # Try to start queued jobs in priority order
        started_jobs = []
        
        for job in self.queued_jobs:
            if self._can_allocate_resources(job):
                if self._start_job(job):
                    started_jobs.append(job)
        
        # Remove started jobs from queue
        for job in started_jobs:
            self.queued_jobs.remove(job)
        
        # Try to restart preempted jobs
        restarted_jobs = []
        for job in self.preempted_jobs:
            if self._can_allocate_resources(job):
                if self._start_job(job):
                    restarted_jobs.append(job)
        
        # Remove restarted jobs from preempted list
        for job in restarted_jobs:
            self.preempted_jobs.remove(job)
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a job"""
        # Check running jobs
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            return {
                "job_id": job_id,
                "status": "running",
                "priority": job.priority.value,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "preempted_count": job.preempted_count
            }
        
        # Check queued jobs
        for job in self.queued_jobs:
            if job.job_id == job_id:
                return {
                    "job_id": job_id,
                    "status": "queued",
                    "priority": job.priority.value,
                    "queue_position": self.queued_jobs.index(job),
                    "preempted_count": job.preempted_count
                }
        
        # Check preempted jobs
        for job in self.preempted_jobs:
            if job.job_id == job_id:
                return {
                    "job_id": job_id,
                    "status": "preempted",
                    "priority": job.priority.value,
                    "preempted_count": job.preempted_count
                }
        
        return None
    
    def get_cluster_status(self) -> Dict:
        """Get overall cluster status"""
        return {
            "total_gpu_memory": self.total_gpu_memory,
            "available_gpu_memory": self.available_gpu_memory,
            "utilization": (self.total_gpu_memory - self.available_gpu_memory) / self.total_gpu_memory,
            "running_jobs": len(self.running_jobs),
            "queued_jobs": len(self.queued_jobs),
            "preempted_jobs": len(self.preempted_jobs),
            "gpu_count": self.gpu_count
        }
    
    def get_priority_stats(self) -> Dict:
        """Get statistics by priority level"""
        stats = {}
        
        for priority in JobPriority:
            running_count = len([j for j in self.running_jobs.values() if j.priority == priority])
            queued_count = len([j for j in self.queued_jobs if j.priority == priority])
            preempted_count = len([j for j in self.preempted_jobs if j.priority == priority])
            
            stats[priority.value] = {
                "running": running_count,
                "queued": queued_count,
                "preempted": preempted_count,
                "total": running_count + queued_count + preempted_count
            }
        
        return stats

# Global preemption policy service
preemption_policy_service = PreemptionPolicyService()
