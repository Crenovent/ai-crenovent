"""
Override Service - Manual override capabilities for ML nodes
Task 4.1.6: Add override hooks for ML nodes (manual justifications)
"""

import asyncio
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sqlite3
from pathlib import Path

from .base import BaseOperator, OperatorContext, OperatorResult

logger = logging.getLogger(__name__)

@dataclass
class OverrideRequest:
    """Override request for ML node"""
    override_id: str
    workflow_id: str
    step_id: str
    model_id: str
    node_id: str
    original_prediction: Any
    override_prediction: Any
    justification: str
    override_type: str  # 'manual', 'policy', 'compliance', 'business_rule'
    requested_by: int  # user_id
    tenant_id: str
    status: str = "pending"  # 'pending', 'approved', 'rejected', 'expired'
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

@dataclass
class OverrideApproval:
    """Override approval record"""
    approval_id: str
    override_id: str
    approver_id: int
    approval_status: str  # 'approved', 'rejected'
    approval_reason: str
    approved_at: datetime = field(default_factory=datetime.utcnow)
    tenant_id: str = ""

@dataclass
class OverrideLedger:
    """Immutable override ledger entry"""
    ledger_id: str
    override_id: str
    workflow_id: str
    step_id: str
    model_id: str
    node_id: str
    original_prediction: Any
    override_prediction: Any
    justification: str
    override_type: str
    requested_by: int
    approved_by: Optional[int]
    approval_status: str
    approval_reason: str
    hash_chain: str  # Previous hash for chaining
    current_hash: str  # Current entry hash
    tenant_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)

class OverrideService:
    """
    Service for managing manual overrides of ML node predictions
    
    Provides:
    - Override request management
    - Approval workflow
    - Immutable override ledger
    - Hash-chained audit trail
    - Integration with governance systems
    """
    
    def __init__(self, db_path: str = "override_service.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the override service database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create override requests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS override_requests (
                    override_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    original_prediction TEXT NOT NULL,  -- JSON serialized
                    override_prediction TEXT NOT NULL,  -- JSON serialized
                    justification TEXT NOT NULL,
                    override_type TEXT NOT NULL,
                    requested_by INTEGER NOT NULL,
                    tenant_id TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')
            
            # Create override approvals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS override_approvals (
                    approval_id TEXT PRIMARY KEY,
                    override_id TEXT NOT NULL,
                    approver_id INTEGER NOT NULL,
                    approval_status TEXT NOT NULL,
                    approval_reason TEXT NOT NULL,
                    approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tenant_id TEXT NOT NULL,
                    FOREIGN KEY (override_id) REFERENCES override_requests (override_id)
                )
            ''')
            
            # Create override ledger table (immutable)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS override_ledger (
                    ledger_id TEXT PRIMARY KEY,
                    override_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    original_prediction TEXT NOT NULL,  -- JSON serialized
                    override_prediction TEXT NOT NULL,  -- JSON serialized
                    justification TEXT NOT NULL,
                    override_type TEXT NOT NULL,
                    requested_by INTEGER NOT NULL,
                    approved_by INTEGER,
                    approval_status TEXT NOT NULL,
                    approval_reason TEXT NOT NULL,
                    hash_chain TEXT NOT NULL,
                    current_hash TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_override_requests_workflow ON override_requests(workflow_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_override_requests_tenant ON override_requests(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_override_requests_status ON override_requests(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_override_ledger_workflow ON override_ledger(workflow_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_override_ledger_tenant ON override_ledger(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_override_ledger_hash ON override_ledger(current_hash)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Override service database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize override database: {e}")
            raise
    
    async def create_override_request(
        self,
        workflow_id: str,
        step_id: str,
        model_id: str,
        node_id: str,
        original_prediction: Any,
        override_prediction: Any,
        justification: str,
        override_type: str,
        requested_by: int,
        tenant_id: str,
        expires_in_hours: int = 24
    ) -> str:
        """Create a new override request"""
        try:
            override_id = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO override_requests (
                    override_id, workflow_id, step_id, model_id, node_id,
                    original_prediction, override_prediction, justification,
                    override_type, requested_by, tenant_id, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                override_id,
                workflow_id,
                step_id,
                model_id,
                node_id,
                json.dumps(original_prediction),
                json.dumps(override_prediction),
                justification,
                override_type,
                requested_by,
                tenant_id,
                expires_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created override request: {override_id}")
            return override_id
            
        except Exception as e:
            self.logger.error(f"Failed to create override request: {e}")
            raise
    
    async def get_override_request(self, override_id: str) -> Optional[OverrideRequest]:
        """Get override request by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT override_id, workflow_id, step_id, model_id, node_id,
                       original_prediction, override_prediction, justification,
                       override_type, requested_by, tenant_id, status,
                       created_at, expires_at
                FROM override_requests 
                WHERE override_id = ?
            ''', (override_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return OverrideRequest(
                override_id=row[0],
                workflow_id=row[1],
                step_id=row[2],
                model_id=row[3],
                node_id=row[4],
                original_prediction=json.loads(row[5]),
                override_prediction=json.loads(row[6]),
                justification=row[7],
                override_type=row[8],
                requested_by=row[9],
                tenant_id=row[10],
                status=row[11],
                created_at=datetime.fromisoformat(row[12]),
                expires_at=datetime.fromisoformat(row[13]) if row[13] else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get override request {override_id}: {e}")
            return None
    
    async def get_pending_overrides(
        self,
        tenant_id: str,
        approver_id: Optional[int] = None,
        limit: int = 100
    ) -> List[OverrideRequest]:
        """Get pending override requests"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT override_id, workflow_id, step_id, model_id, node_id,
                       original_prediction, override_prediction, justification,
                       override_type, requested_by, tenant_id, status,
                       created_at, expires_at
                FROM override_requests 
                WHERE tenant_id = ? AND status = 'pending'
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            '''
            params = [tenant_id]
            
            if approver_id:
                # Add approver-specific filtering logic here
                pass
            
            query += ' ORDER BY created_at ASC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            overrides = []
            for row in rows:
                overrides.append(OverrideRequest(
                    override_id=row[0],
                    workflow_id=row[1],
                    step_id=row[2],
                    model_id=row[3],
                    node_id=row[4],
                    original_prediction=json.loads(row[5]),
                    override_prediction=json.loads(row[6]),
                    justification=row[7],
                    override_type=row[8],
                    requested_by=row[9],
                    tenant_id=row[10],
                    status=row[11],
                    created_at=datetime.fromisoformat(row[12]),
                    expires_at=datetime.fromisoformat(row[13]) if row[13] else None
                ))
            
            return overrides
            
        except Exception as e:
            self.logger.error(f"Failed to get pending overrides: {e}")
            return []
    
    async def approve_override(
        self,
        override_id: str,
        approver_id: int,
        approval_status: str,
        approval_reason: str,
        tenant_id: str
    ) -> bool:
        """Approve or reject an override request"""
        try:
            # Get the override request
            override_request = await self.get_override_request(override_id)
            if not override_request:
                return False
            
            if override_request.status != "pending":
                self.logger.warning(f"Override {override_id} is not pending")
                return False
            
            # Check if expired
            if override_request.expires_at and override_request.expires_at < datetime.utcnow():
                await self._expire_override(override_id)
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create approval record
            approval_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO override_approvals (
                    approval_id, override_id, approver_id, approval_status,
                    approval_reason, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                approval_id,
                override_id,
                approver_id,
                approval_status,
                approval_reason,
                tenant_id
            ))
            
            # Update override request status
            cursor.execute('''
                UPDATE override_requests 
                SET status = ? 
                WHERE override_id = ?
            ''', (approval_status, override_id))
            
            # Add to immutable ledger
            await self._add_to_ledger(override_request, approver_id, approval_status, approval_reason, tenant_id)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Override {override_id} {approval_status} by {approver_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to approve override {override_id}: {e}")
            return False
    
    async def _expire_override(self, override_id: str):
        """Mark an override as expired"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE override_requests 
                SET status = 'expired' 
                WHERE override_id = ?
            ''', (override_id,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Override {override_id} expired")
            
        except Exception as e:
            self.logger.error(f"Failed to expire override {override_id}: {e}")
    
    async def _add_to_ledger(
        self,
        override_request: OverrideRequest,
        approver_id: int,
        approval_status: str,
        approval_reason: str,
        tenant_id: str
    ):
        """Add override to immutable ledger with hash chaining"""
        try:
            # Get the last ledger entry for hash chaining
            last_hash = await self._get_last_ledger_hash(tenant_id)
            
            # Create ledger entry
            ledger_id = str(uuid.uuid4())
            ledger_entry = OverrideLedger(
                ledger_id=ledger_id,
                override_id=override_request.override_id,
                workflow_id=override_request.workflow_id,
                step_id=override_request.step_id,
                model_id=override_request.model_id,
                node_id=override_request.node_id,
                original_prediction=override_request.original_prediction,
                override_prediction=override_request.override_prediction,
                justification=override_request.justification,
                override_type=override_request.override_type,
                requested_by=override_request.requested_by,
                approved_by=approver_id,
                approval_status=approval_status,
                approval_reason=approval_reason,
                hash_chain=last_hash,
                current_hash="",  # Will be calculated
                tenant_id=tenant_id
            )
            
            # Calculate hash
            current_hash = self._calculate_ledger_hash(ledger_entry, last_hash)
            ledger_entry.current_hash = current_hash
            
            # Insert into ledger
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO override_ledger (
                    ledger_id, override_id, workflow_id, step_id, model_id, node_id,
                    original_prediction, override_prediction, justification, override_type,
                    requested_by, approved_by, approval_status, approval_reason,
                    hash_chain, current_hash, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ledger_entry.ledger_id,
                ledger_entry.override_id,
                ledger_entry.workflow_id,
                ledger_entry.step_id,
                ledger_entry.model_id,
                ledger_entry.node_id,
                json.dumps(ledger_entry.original_prediction),
                json.dumps(ledger_entry.override_prediction),
                ledger_entry.justification,
                ledger_entry.override_type,
                ledger_entry.requested_by,
                ledger_entry.approved_by,
                ledger_entry.approval_status,
                ledger_entry.approval_reason,
                ledger_entry.hash_chain,
                ledger_entry.current_hash,
                ledger_entry.tenant_id
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added override to ledger: {ledger_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add override to ledger: {e}")
            raise
    
    async def _get_last_ledger_hash(self, tenant_id: str) -> str:
        """Get the last ledger entry hash for chaining"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT current_hash 
                FROM override_ledger 
                WHERE tenant_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (tenant_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            return row[0] if row else "genesis"
            
        except Exception as e:
            self.logger.error(f"Failed to get last ledger hash: {e}")
            return "genesis"
    
    def _calculate_ledger_hash(self, ledger_entry: OverrideLedger, previous_hash: str) -> str:
        """Calculate hash for ledger entry"""
        try:
            # Create hash input
            hash_input = {
                "ledger_id": ledger_entry.ledger_id,
                "override_id": ledger_entry.override_id,
                "workflow_id": ledger_entry.workflow_id,
                "step_id": ledger_entry.step_id,
                "model_id": ledger_entry.model_id,
                "node_id": ledger_entry.node_id,
                "original_prediction": ledger_entry.original_prediction,
                "override_prediction": ledger_entry.override_prediction,
                "justification": ledger_entry.justification,
                "override_type": ledger_entry.override_type,
                "requested_by": ledger_entry.requested_by,
                "approved_by": ledger_entry.approved_by,
                "approval_status": ledger_entry.approval_status,
                "approval_reason": ledger_entry.approval_reason,
                "previous_hash": previous_hash,
                "timestamp": ledger_entry.created_at.isoformat()
            }
            
            # Create hash
            hash_string = json.dumps(hash_input, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate ledger hash: {e}")
            return "error"
    
    async def get_override_ledger(
        self,
        tenant_id: str,
        workflow_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[OverrideLedger]:
        """Get override ledger entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT ledger_id, override_id, workflow_id, step_id, model_id, node_id,
                       original_prediction, override_prediction, justification, override_type,
                       requested_by, approved_by, approval_status, approval_reason,
                       hash_chain, current_hash, tenant_id, created_at
                FROM override_ledger 
                WHERE tenant_id = ?
            '''
            params = [tenant_id]
            
            if workflow_id:
                query += ' AND workflow_id = ?'
                params.append(workflow_id)
            
            if start_date:
                query += ' AND created_at >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND created_at <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY created_at ASC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            ledger_entries = []
            for row in rows:
                ledger_entries.append(OverrideLedger(
                    ledger_id=row[0],
                    override_id=row[1],
                    workflow_id=row[2],
                    step_id=row[3],
                    model_id=row[4],
                    node_id=row[5],
                    original_prediction=json.loads(row[6]),
                    override_prediction=json.loads(row[7]),
                    justification=row[8],
                    override_type=row[9],
                    requested_by=row[10],
                    approved_by=row[11],
                    approval_status=row[12],
                    approval_reason=row[13],
                    hash_chain=row[14],
                    current_hash=row[15],
                    tenant_id=row[16],
                    created_at=datetime.fromisoformat(row[17])
                ))
            
            return ledger_entries
            
        except Exception as e:
            self.logger.error(f"Failed to get override ledger: {e}")
            return []
    
    async def verify_ledger_integrity(self, tenant_id: str) -> Dict[str, Any]:
        """Verify the integrity of the override ledger"""
        try:
            ledger_entries = await self.get_override_ledger(tenant_id, limit=1000)
            
            if not ledger_entries:
                return {"integrity_check": "passed", "message": "No entries to verify"}
            
            # Verify hash chain
            previous_hash = "genesis"
            integrity_issues = []
            
            for i, entry in enumerate(ledger_entries):
                # Check if hash chain is correct
                if entry.hash_chain != previous_hash:
                    integrity_issues.append(f"Hash chain broken at entry {i}: {entry.ledger_id}")
                
                # Verify current hash
                expected_hash = self._calculate_ledger_hash(entry, previous_hash)
                if entry.current_hash != expected_hash:
                    integrity_issues.append(f"Hash mismatch at entry {i}: {entry.ledger_id}")
                
                previous_hash = entry.current_hash
            
            return {
                "integrity_check": "passed" if not integrity_issues else "failed",
                "total_entries": len(ledger_entries),
                "issues": integrity_issues,
                "last_hash": previous_hash
            }
            
        except Exception as e:
            self.logger.error(f"Failed to verify ledger integrity: {e}")
            return {"integrity_check": "error", "message": str(e)}
    
    async def get_override_analytics(
        self,
        tenant_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get override analytics and metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get override statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_overrides,
                    SUM(CASE WHEN approval_status = 'approved' THEN 1 ELSE 0 END) as approved_overrides,
                    SUM(CASE WHEN approval_status = 'rejected' THEN 1 ELSE 0 END) as rejected_overrides,
                    override_type,
                    model_id
                FROM override_ledger 
                WHERE tenant_id = ? AND created_at >= datetime('now', '-{} days')
                GROUP BY override_type, model_id
            '''.format(days), (tenant_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            analytics = {
                "total_overrides": sum(row[0] for row in rows),
                "approved_overrides": sum(row[1] for row in rows),
                "rejected_overrides": sum(row[2] for row in rows),
                "approval_rate": 0,
                "by_override_type": {},
                "by_model": {},
                "days_analyzed": days
            }
            
            if analytics["total_overrides"] > 0:
                analytics["approval_rate"] = analytics["approved_overrides"] / analytics["total_overrides"]
            
            # Group by override type
            for row in rows:
                override_type = row[3]
                if override_type not in analytics["by_override_type"]:
                    analytics["by_override_type"][override_type] = {
                        "total": 0,
                        "approved": 0,
                        "rejected": 0
                    }
                analytics["by_override_type"][override_type]["total"] += row[0]
                analytics["by_override_type"][override_type]["approved"] += row[1]
                analytics["by_override_type"][override_type]["rejected"] += row[2]
            
            # Group by model
            for row in rows:
                model_id = row[4]
                if model_id not in analytics["by_model"]:
                    analytics["by_model"][model_id] = {
                        "total": 0,
                        "approved": 0,
                        "rejected": 0
                    }
                analytics["by_model"][model_id]["total"] += row[0]
                analytics["by_model"][model_id]["approved"] += row[1]
                analytics["by_model"][model_id]["rejected"] += row[2]
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get override analytics: {e}")
            return {}
    
    async def cleanup_expired_overrides(self) -> int:
        """Clean up expired override requests"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE override_requests 
                SET status = 'expired' 
                WHERE status = 'pending' 
                AND expires_at IS NOT NULL 
                AND expires_at < CURRENT_TIMESTAMP
            ''')
            
            expired_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Expired {expired_count} override requests")
            return expired_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired overrides: {e}")
            return 0
