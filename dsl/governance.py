"""
Governance Engine - Manages policies, compliance, and audit trails
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PolicyEngine:
    """
    Policy engine for managing governance policies and compliance rules
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        self._policy_cache = {}
    
    async def load_policy_pack(self, policy_pack_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Load policy pack from database"""
        try:
            # Check cache first
            cache_key = f"{tenant_id}:{policy_pack_id}"
            if cache_key in self._policy_cache:
                return self._policy_cache[cache_key]
            
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return None
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    SELECT 
                        policy_pack_id, name, description, version,
                        rules, industry, compliance_standards,
                        is_global, created_at, updated_at
                    FROM dsl_policy_packs
                    WHERE policy_pack_id = $1
                    AND (is_global = true OR tenant_id = $2)
                """
                
                row = await conn.fetchrow(query, policy_pack_id, tenant_id)
                
                if row:
                    policy_pack = {
                        'policy_pack_id': row['policy_pack_id'],
                        'name': row['name'],
                        'description': row['description'],
                        'version': row['version'],
                        'rules': row['rules'],
                        'industry': row['industry'],
                        'compliance_standards': row['compliance_standards'],
                        'is_global': row['is_global'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None
                    }
                    
                    # Cache the policy pack
                    self._policy_cache[cache_key] = policy_pack
                    return policy_pack
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading policy pack {policy_pack_id}: {e}")
            return None
    
    async def create_policy_pack(
        self,
        policy_pack_id: str,
        name: str,
        rules: List[Dict[str, Any]],
        tenant_id: str,
        created_by_user_id: int,
        description: str = "",
        industry: str = "SaaS",
        compliance_standards: List[str] = None,
        is_global: bool = False
    ) -> bool:
        """Create a new policy pack"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return False
            
            compliance_standards = compliance_standards or []
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                if not is_global:
                    await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    INSERT INTO dsl_policy_packs (
                        policy_pack_id, name, description, rules,
                        industry, compliance_standards, tenant_id,
                        is_global, created_by_user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """
                
                await conn.execute(
                    query,
                    policy_pack_id,
                    name,
                    description,
                    json.dumps(rules),
                    industry,
                    json.dumps(compliance_standards),
                    tenant_id if not is_global else None,
                    is_global,
                    created_by_user_id
                )
                
                self.logger.info(f"Policy pack created: {policy_pack_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating policy pack: {e}")
            return False
    
    async def validate_against_policy(
        self,
        policy_pack_id: str,
        tenant_id: str,
        data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Validate data against policy pack"""
        try:
            policy_pack = await self.load_policy_pack(policy_pack_id, tenant_id)
            if not policy_pack:
                return {
                    'valid': False,
                    'violations': [f'Policy pack not found: {policy_pack_id}']
                }
            
            violations = []
            rules_checked = 0
            
            for rule in policy_pack['rules']:
                rules_checked += 1
                violation = await self._check_rule(rule, data, context or {})
                if violation:
                    violations.append(violation)
            
            return {
                'valid': len(violations) == 0,
                'violations': violations,
                'rules_checked': rules_checked,
                'policy_pack': policy_pack['name']
            }
            
        except Exception as e:
            self.logger.error(f"Error validating against policy: {e}")
            return {
                'valid': False,
                'violations': [f'Policy validation error: {e}']
            }
    
    async def _check_rule(self, rule: Dict[str, Any], data: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Check a single policy rule"""
        try:
            rule_type = rule.get('type')
            mandatory = rule.get('mandatory', False)
            
            if rule_type == 'evidence_required':
                if mandatory and not context.get('evidence_capture', False):
                    return f"Evidence is required: {rule.get('description', 'Evidence capture mandatory')}"
            
            elif rule_type == 'approval_required':
                condition = rule.get('condition')
                if condition and self._evaluate_condition(condition, data):
                    approver_role = rule.get('approver_role', 'Manager')
                    if not context.get('approved_by'):
                        return f"Approval required from {approver_role}: {rule.get('description', 'Approval needed')}"
            
            elif rule_type == 'data_minimization':
                allowed_fields = rule.get('allowed_fields', [])
                if allowed_fields:
                    for field in data.keys():
                        if field not in allowed_fields:
                            return f"Field '{field}' not allowed by data minimization policy"
            
            elif rule_type == 'data_retention':
                # This would be checked during data cleanup processes
                pass
            
            elif rule_type == 'consent_required':
                if mandatory and not context.get('consent_given', False):
                    return f"Consent is required: {rule.get('description', 'User consent mandatory')}"
            
            elif rule_type == 'field_required':
                required_fields = rule.get('fields', [])
                for field in required_fields:
                    if field not in data or not data[field]:
                        return f"Required field missing: {field}"
            
            elif rule_type == 'value_range':
                field = rule.get('field')
                min_value = rule.get('min_value')
                max_value = rule.get('max_value')
                
                if field in data:
                    value = data[field]
                    try:
                        numeric_value = float(value)
                        if min_value is not None and numeric_value < min_value:
                            return f"Value {value} below minimum {min_value} for field {field}"
                        if max_value is not None and numeric_value > max_value:
                            return f"Value {value} above maximum {max_value} for field {field}"
                    except (ValueError, TypeError):
                        if mandatory:
                            return f"Field {field} must be numeric for range validation"
            
            return None
            
        except Exception as e:
            return f"Rule check error: {e}"
    
    def _evaluate_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """Safely evaluate policy condition"""
        try:
            # Replace field references with actual values
            for field_name, field_value in data.items():
                if isinstance(field_value, str):
                    condition = condition.replace(field_name, f"'{field_value}'")
                else:
                    condition = condition.replace(field_name, str(field_value))
            
            # Basic evaluation (in production, use a safer evaluation method)
            return eval(condition)
            
        except Exception as e:
            self.logger.warning(f"Condition evaluation error: {e}")
            return False

class EvidenceManager:
    """
    Evidence manager for audit trails and compliance documentation
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
    
    async def create_evidence_pack(
        self,
        trace_id: str,
        evidence_data: Dict[str, Any],
        tenant_id: str,
        compliance_status: str = "compliant"
    ) -> Optional[str]:
        """Create an evidence pack for audit trail"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return None
            
            # Generate evidence hash
            import hashlib
            evidence_json = json.dumps(evidence_data, sort_keys=True, separators=(',', ':'))
            evidence_hash = hashlib.sha256(evidence_json.encode('utf-8')).hexdigest()
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    INSERT INTO dsl_evidence_packs (
                        trace_id, evidence_data, evidence_hash,
                        compliance_status, tenant_id
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING evidence_pack_id
                """
                
                row = await conn.fetchrow(
                    query,
                    trace_id,
                    evidence_json,
                    evidence_hash,
                    compliance_status,
                    tenant_id
                )
                
                evidence_pack_id = str(row['evidence_pack_id'])
                self.logger.info(f"Evidence pack created: {evidence_pack_id}")
                return evidence_pack_id
                
        except Exception as e:
            self.logger.error(f"Error creating evidence pack: {e}")
            return None
    
    async def get_evidence_pack(self, evidence_pack_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evidence pack by ID"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return None
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    SELECT 
                        evidence_pack_id, trace_id, evidence_data,
                        evidence_hash, policy_violations, compliance_status,
                        blob_storage_url, created_at
                    FROM dsl_evidence_packs
                    WHERE evidence_pack_id = $1 AND tenant_id = $2
                """
                
                row = await conn.fetchrow(query, evidence_pack_id, tenant_id)
                
                if row:
                    return {
                        'evidence_pack_id': str(row['evidence_pack_id']),
                        'trace_id': row['trace_id'],
                        'evidence_data': json.loads(row['evidence_data']) if row['evidence_data'] else {},
                        'evidence_hash': row['evidence_hash'],
                        'policy_violations': row['policy_violations'] or [],
                        'compliance_status': row['compliance_status'],
                        'blob_storage_url': row['blob_storage_url'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving evidence pack: {e}")
            return None
    
    async def search_evidence_packs(
        self,
        tenant_id: str,
        trace_id: Optional[str] = None,
        compliance_status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search evidence packs with filters"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return []
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                # Build query with filters
                where_conditions = ["tenant_id = $1"]
                params = [tenant_id]
                param_count = 2
                
                if trace_id:
                    where_conditions.append(f"trace_id = ${param_count}")
                    params.append(trace_id)
                    param_count += 1
                
                if compliance_status:
                    where_conditions.append(f"compliance_status = ${param_count}")
                    params.append(compliance_status)
                    param_count += 1
                
                if start_date:
                    where_conditions.append(f"created_at >= ${param_count}")
                    params.append(start_date)
                    param_count += 1
                
                if end_date:
                    where_conditions.append(f"created_at <= ${param_count}")
                    params.append(end_date)
                    param_count += 1
                
                where_clause = " AND ".join(where_conditions)
                
                query = f"""
                    SELECT 
                        evidence_pack_id, trace_id, evidence_hash,
                        policy_violations, compliance_status, created_at
                    FROM dsl_evidence_packs
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT {limit}
                """
                
                rows = await conn.fetch(query, *params)
                
                evidence_packs = []
                for row in rows:
                    evidence_packs.append({
                        'evidence_pack_id': str(row['evidence_pack_id']),
                        'trace_id': row['trace_id'],
                        'evidence_hash': row['evidence_hash'],
                        'policy_violations': row['policy_violations'] or [],
                        'compliance_status': row['compliance_status'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None
                    })
                
                return evidence_packs
                
        except Exception as e:
            self.logger.error(f"Error searching evidence packs: {e}")
            return []
    
    async def record_override(
        self,
        trace_id: str,
        step_id: str,
        original_decision: Dict[str, Any],
        override_decision: Dict[str, Any],
        override_reason: str,
        override_by_user_id: int,
        tenant_id: str,
        requires_approval: bool = False
    ) -> Optional[str]:
        """Record a human override for audit trail"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return None
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                query = """
                    INSERT INTO dsl_override_ledger (
                        trace_id, step_id, original_decision, override_decision,
                        override_reason, override_by_user_id, tenant_id,
                        requires_approval
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING override_id
                """
                
                row = await conn.fetchrow(
                    query,
                    trace_id,
                    step_id,
                    json.dumps(original_decision),
                    json.dumps(override_decision),
                    override_reason,
                    override_by_user_id,
                    tenant_id,
                    requires_approval
                )
                
                override_id = str(row['override_id'])
                self.logger.info(f"Override recorded: {override_id}")
                return override_id
                
        except Exception as e:
            self.logger.error(f"Error recording override: {e}")
            return None
    
    async def get_override_history(
        self,
        tenant_id: str,
        trace_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get override history with optional filters"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return []
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                # Build query with filters
                where_conditions = ["tenant_id = $1"]
                params = [tenant_id]
                param_count = 2
                
                if trace_id:
                    where_conditions.append(f"trace_id = ${param_count}")
                    params.append(trace_id)
                    param_count += 1
                
                if user_id:
                    where_conditions.append(f"override_by_user_id = ${param_count}")
                    params.append(user_id)
                    param_count += 1
                
                where_clause = " AND ".join(where_conditions)
                
                query = f"""
                    SELECT 
                        override_id, trace_id, step_id, override_reason,
                        override_by_user_id, override_at, requires_approval,
                        approved_by_user_id, approved_at, approval_status
                    FROM dsl_override_ledger
                    WHERE {where_clause}
                    ORDER BY override_at DESC
                    LIMIT {limit}
                """
                
                rows = await conn.fetch(query, *params)
                
                overrides = []
                for row in rows:
                    overrides.append({
                        'override_id': str(row['override_id']),
                        'trace_id': row['trace_id'],
                        'step_id': row['step_id'],
                        'override_reason': row['override_reason'],
                        'override_by_user_id': row['override_by_user_id'],
                        'override_at': row['override_at'].isoformat() if row['override_at'] else None,
                        'requires_approval': row['requires_approval'],
                        'approved_by_user_id': row['approved_by_user_id'],
                        'approved_at': row['approved_at'].isoformat() if row['approved_at'] else None,
                        'approval_status': row['approval_status']
                    })
                
                return overrides
                
        except Exception as e:
            self.logger.error(f"Error getting override history: {e}")
            return []
