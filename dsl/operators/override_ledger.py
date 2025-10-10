"""
Override Ledger - Immutable ledger for override tracking
Task 4.1.6: Add override hooks for ML nodes (manual justifications)
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LedgerEntry:
    """Immutable ledger entry for overrides"""
    entry_id: str
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
    previous_hash: str
    current_hash: str
    tenant_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class LedgerIntegrityReport:
    """Ledger integrity verification report"""
    is_valid: bool
    total_entries: int
    verified_entries: int
    broken_chain_positions: List[int]
    hash_mismatches: List[int]
    last_valid_hash: str
    verification_timestamp: datetime = field(default_factory=datetime.utcnow)

class OverrideLedger:
    """
    Immutable ledger for tracking all override operations
    
    Provides:
    - Immutable audit trail
    - Hash-chained integrity verification
    - Cryptographic tamper detection
    - Compliance-ready audit records
    - Historical query capabilities
    """
    
    def __init__(self, db_path: str = "override_ledger.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the override ledger database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create immutable ledger table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS override_ledger (
                    entry_id TEXT PRIMARY KEY,
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
                    previous_hash TEXT NOT NULL,
                    current_hash TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_workflow ON override_ledger(workflow_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_tenant ON override_ledger(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_created_at ON override_ledger(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_hash ON override_ledger(current_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_override_id ON override_ledger(override_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_model ON override_ledger(model_id)')
            
            # Create integrity verification table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS integrity_checks (
                    check_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_valid BOOLEAN NOT NULL,
                    total_entries INTEGER NOT NULL,
                    verified_entries INTEGER NOT NULL,
                    broken_chain_positions TEXT,  -- JSON array
                    hash_mismatches TEXT,  -- JSON array
                    last_valid_hash TEXT,
                    check_duration_ms INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Override ledger database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize override ledger database: {e}")
            raise
    
    async def add_ledger_entry(
        self,
        override_id: str,
        workflow_id: str,
        step_id: str,
        model_id: str,
        node_id: str,
        original_prediction: Any,
        override_prediction: Any,
        justification: str,
        override_type: str,
        requested_by: int,
        approved_by: Optional[int],
        approval_status: str,
        approval_reason: str,
        tenant_id: str
    ) -> str:
        """Add a new entry to the immutable ledger"""
        try:
            entry_id = str(uuid.uuid4())
            
            # Get the last hash for chaining
            previous_hash = await self._get_last_hash(tenant_id)
            
            # Create ledger entry
            ledger_entry = LedgerEntry(
                entry_id=entry_id,
                override_id=override_id,
                workflow_id=workflow_id,
                step_id=step_id,
                model_id=model_id,
                node_id=node_id,
                original_prediction=original_prediction,
                override_prediction=override_prediction,
                justification=justification,
                override_type=override_type,
                requested_by=requested_by,
                approved_by=approved_by,
                approval_status=approval_status,
                approval_reason=approval_reason,
                previous_hash=previous_hash,
                current_hash="",  # Will be calculated
                tenant_id=tenant_id
            )
            
            # Calculate hash
            current_hash = self._calculate_entry_hash(ledger_entry)
            ledger_entry.current_hash = current_hash
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO override_ledger (
                    entry_id, override_id, workflow_id, step_id, model_id, node_id,
                    original_prediction, override_prediction, justification, override_type,
                    requested_by, approved_by, approval_status, approval_reason,
                    previous_hash, current_hash, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ledger_entry.entry_id,
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
                ledger_entry.previous_hash,
                ledger_entry.current_hash,
                ledger_entry.tenant_id
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added ledger entry: {entry_id}")
            return entry_id
            
        except Exception as e:
            self.logger.error(f"Failed to add ledger entry: {e}")
            raise
    
    async def get_ledger_entries(
        self,
        tenant_id: str,
        workflow_id: Optional[str] = None,
        model_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[LedgerEntry]:
        """Get ledger entries with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT entry_id, override_id, workflow_id, step_id, model_id, node_id,
                       original_prediction, override_prediction, justification, override_type,
                       requested_by, approved_by, approval_status, approval_reason,
                       previous_hash, current_hash, tenant_id, created_at
                FROM override_ledger 
                WHERE tenant_id = ?
            '''
            params = [tenant_id]
            
            if workflow_id:
                query += ' AND workflow_id = ?'
                params.append(workflow_id)
            
            if model_id:
                query += ' AND model_id = ?'
                params.append(model_id)
            
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
            
            entries = []
            for row in rows:
                entries.append(LedgerEntry(
                    entry_id=row[0],
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
                    previous_hash=row[14],
                    current_hash=row[15],
                    tenant_id=row[16],
                    created_at=datetime.fromisoformat(row[17])
                ))
            
            return entries
            
        except Exception as e:
            self.logger.error(f"Failed to get ledger entries: {e}")
            return []
    
    async def verify_ledger_integrity(self, tenant_id: str) -> LedgerIntegrityReport:
        """Verify the integrity of the ledger using hash chaining"""
        try:
            start_time = datetime.utcnow()
            
            # Get all entries for the tenant
            entries = await self.get_ledger_entries(tenant_id, limit=10000)
            
            if not entries:
                return LedgerIntegrityReport(
                    is_valid=True,
                    total_entries=0,
                    verified_entries=0,
                    broken_chain_positions=[],
                    hash_mismatches=[],
                    last_valid_hash="genesis"
                )
            
            # Verify hash chain
            previous_hash = "genesis"
            verified_entries = 0
            broken_chain_positions = []
            hash_mismatches = []
            
            for i, entry in enumerate(entries):
                # Check if previous hash matches
                if entry.previous_hash != previous_hash:
                    broken_chain_positions.append(i)
                    continue
                
                # Verify current hash
                expected_hash = self._calculate_entry_hash(entry)
                if entry.current_hash != expected_hash:
                    hash_mismatches.append(i)
                    continue
                
                verified_entries += 1
                previous_hash = entry.current_hash
            
            # Calculate verification duration
            verification_duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Create integrity report
            report = LedgerIntegrityReport(
                is_valid=len(broken_chain_positions) == 0 and len(hash_mismatches) == 0,
                total_entries=len(entries),
                verified_entries=verified_entries,
                broken_chain_positions=broken_chain_positions,
                hash_mismatches=hash_mismatches,
                last_valid_hash=previous_hash
            )
            
            # Log integrity check
            await self._log_integrity_check(tenant_id, report, verification_duration)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to verify ledger integrity: {e}")
            return LedgerIntegrityReport(
                is_valid=False,
                total_entries=0,
                verified_entries=0,
                broken_chain_positions=[],
                hash_mismatches=[],
                last_valid_hash="error"
            )
    
    def _calculate_entry_hash(self, entry: LedgerEntry) -> str:
        """Calculate SHA-256 hash for a ledger entry"""
        try:
            # Create hash input (exclude current_hash to avoid circular reference)
            hash_input = {
                "entry_id": entry.entry_id,
                "override_id": entry.override_id,
                "workflow_id": entry.workflow_id,
                "step_id": entry.step_id,
                "model_id": entry.model_id,
                "node_id": entry.node_id,
                "original_prediction": entry.original_prediction,
                "override_prediction": entry.override_prediction,
                "justification": entry.justification,
                "override_type": entry.override_type,
                "requested_by": entry.requested_by,
                "approved_by": entry.approved_by,
                "approval_status": entry.approval_status,
                "approval_reason": entry.approval_reason,
                "previous_hash": entry.previous_hash,
                "tenant_id": entry.tenant_id,
                "timestamp": entry.created_at.isoformat()
            }
            
            # Create hash
            hash_string = json.dumps(hash_input, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate entry hash: {e}")
            return "error"
    
    async def _get_last_hash(self, tenant_id: str) -> str:
        """Get the last hash from the ledger for chaining"""
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
            self.logger.error(f"Failed to get last hash: {e}")
            return "genesis"
    
    async def _log_integrity_check(
        self, 
        tenant_id: str, 
        report: LedgerIntegrityReport, 
        duration_ms: int
    ):
        """Log integrity check results"""
        try:
            check_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO integrity_checks (
                    check_id, tenant_id, is_valid, total_entries, verified_entries,
                    broken_chain_positions, hash_mismatches, last_valid_hash, check_duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                check_id,
                tenant_id,
                report.is_valid,
                report.total_entries,
                report.verified_entries,
                json.dumps(report.broken_chain_positions),
                json.dumps(report.hash_mismatches),
                report.last_valid_hash,
                duration_ms
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Logged integrity check: {check_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log integrity check: {e}")
    
    async def get_ledger_analytics(
        self,
        tenant_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get ledger analytics and metrics"""
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
            self.logger.error(f"Failed to get ledger analytics: {e}")
            return {}
    
    async def export_ledger_for_audit(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export ledger entries for audit purposes"""
        try:
            entries = await self.get_ledger_entries(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            if format == "json":
                return {
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "tenant_id": tenant_id,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "total_entries": len(entries),
                    "entries": [
                        {
                            "entry_id": entry.entry_id,
                            "override_id": entry.override_id,
                            "workflow_id": entry.workflow_id,
                            "step_id": entry.step_id,
                            "model_id": entry.model_id,
                            "node_id": entry.node_id,
                            "original_prediction": entry.original_prediction,
                            "override_prediction": entry.override_prediction,
                            "justification": entry.justification,
                            "override_type": entry.override_type,
                            "requested_by": entry.requested_by,
                            "approved_by": entry.approved_by,
                            "approval_status": entry.approval_status,
                            "approval_reason": entry.approval_reason,
                            "previous_hash": entry.previous_hash,
                            "current_hash": entry.current_hash,
                            "created_at": entry.created_at.isoformat()
                        }
                        for entry in entries
                    ]
                }
            elif format == "csv":
                # Convert to CSV format
                csv_data = "entry_id,override_id,workflow_id,step_id,model_id,node_id,original_prediction,override_prediction,justification,override_type,requested_by,approved_by,approval_status,approval_reason,previous_hash,current_hash,created_at\n"
                for entry in entries:
                    csv_data += f"{entry.entry_id},{entry.override_id},{entry.workflow_id},{entry.step_id},{entry.model_id},{entry.node_id},\"{json.dumps(entry.original_prediction)}\",\"{json.dumps(entry.override_prediction)}\",\"{entry.justification}\",{entry.override_type},{entry.requested_by},{entry.approved_by},{entry.approval_status},\"{entry.approval_reason}\",{entry.previous_hash},{entry.current_hash},{entry.created_at.isoformat()}\n"
                return csv_data
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export ledger for audit: {e}")
            return {}
    
    async def get_integrity_check_history(
        self,
        tenant_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get integrity check history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT check_id, check_timestamp, is_valid, total_entries, verified_entries,
                       broken_chain_positions, hash_mismatches, last_valid_hash, check_duration_ms
                FROM integrity_checks 
                WHERE tenant_id = ? 
                ORDER BY check_timestamp DESC 
                LIMIT ?
            ''', (tenant_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            checks = []
            for row in rows:
                checks.append({
                    "check_id": row[0],
                    "check_timestamp": row[1],
                    "is_valid": bool(row[2]),
                    "total_entries": row[3],
                    "verified_entries": row[4],
                    "broken_chain_positions": json.loads(row[5]) if row[5] else [],
                    "hash_mismatches": json.loads(row[6]) if row[6] else [],
                    "last_valid_hash": row[7],
                    "check_duration_ms": row[8]
                })
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Failed to get integrity check history: {e}")
            return []
