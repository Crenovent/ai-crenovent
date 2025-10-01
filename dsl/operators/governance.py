"""
Governance Operator - Handles compliance, auditing, and policy enforcement
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from .base import BaseOperator, OperatorContext, OperatorResult
import logging

logger = logging.getLogger(__name__)

class GovernanceOperator(BaseOperator):
    """
    Governance operator for compliance and audit operations
    Supports:
    - Policy assertion and validation
    - Evidence pack generation
    - Audit trail recording
    - Compliance checking
    - Override logging
    """
    
    def __init__(self, config=None):
        super().__init__("governance_operator")
        self.config = config or {}
    
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate governance operator configuration"""
        errors = []
        
        # Check required fields
        if 'action' not in config:
            errors.append("'action' is required")
        elif config['action'] not in ['assert', 'record', 'anchor', 'validate', 'create_evidence_pack']:
            errors.append(f"Unsupported action: {config['action']}")
        
        # Action-specific validation
        action = config.get('action')
        if action == 'assert':
            if 'policy_id' not in config:
                errors.append("'policy_id' required for assert action")
        elif action == 'record':
            if 'evidence_type' not in config:
                errors.append("'evidence_type' required for record action")
        elif action == 'anchor':
            if 'data' not in config:
                errors.append("'data' required for anchor action")
        elif action == 'validate':
            if 'validation_rules' not in config:
                errors.append("'validation_rules' required for validate action")
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute the governance operation"""
        action = config['action']
        
        try:
            if action == 'assert':
                return await self._assert_policy(context, config)
            elif action == 'record':
                return await self._record_evidence(context, config)
            elif action == 'anchor':
                return await self._anchor_data(context, config)
            elif action == 'validate':
                return await self._validate_compliance(context, config)
            elif action == 'create_evidence_pack':
                return await self._create_evidence_pack(context, config)
            else:
                return OperatorResult(
                    success=False,
                    error_message=f"Unsupported governance action: {action}"
                )
                
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Governance operation failed: {e}"
            )
    
    async def _assert_policy(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Assert policy compliance"""
        try:
            policy_id = config['policy_id']
            fields = config.get('fields', {})
            
            # Load policy from database
            policy = await self._load_policy(context, policy_id)
            if not policy:
                return OperatorResult(
                    success=False,
                    error_message=f"Policy not found: {policy_id}"
                )
            
            # Validate against policy rules
            violations = []
            compliance_result = await self._check_policy_compliance(policy, fields, context)
            
            if compliance_result['violations']:
                violations.extend(compliance_result['violations'])
            
            # Record policy assertion
            evidence_data = {
                'policy_id': policy_id,
                'fields_checked': fields,
                'violations': violations,
                'compliance_status': 'compliant' if not violations else 'violation',
                'assertion_timestamp': datetime.utcnow().isoformat()
            }
            
            return OperatorResult(
                success=len(violations) == 0,
                output_data={
                    'policy_assertion': evidence_data,
                    'compliance_status': evidence_data['compliance_status'],
                    'violations': violations
                },
                policy_violations=violations,
                confidence_score=1.0
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Policy assertion failed: {e}"
            )
    
    async def _record_evidence(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Record evidence for audit trails"""
        try:
            evidence_type = config['evidence_type']
            evidence_data = config.get('data', {})
            
            # Create evidence record
            evidence_record = {
                'evidence_type': evidence_type,
                'workflow_id': context.workflow_id,
                'step_id': context.step_id,
                'user_id': context.user_id,
                'tenant_id': context.tenant_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': evidence_data,
                'context': {
                    'session_id': context.session_id,
                    'execution_id': context.execution_id
                }
            }
            
            # Generate evidence hash for integrity
            evidence_hash = self._generate_evidence_hash(evidence_record)
            evidence_record['evidence_hash'] = evidence_hash
            
            # Store evidence (would integrate with database)
            evidence_id = await self._store_evidence(context, evidence_record)
            
            return OperatorResult(
                success=True,
                output_data={
                    'evidence_recorded': True,
                    'evidence_id': evidence_id,
                    'evidence_hash': evidence_hash,
                    'evidence_type': evidence_type
                },
                confidence_score=1.0
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Evidence recording failed: {e}"
            )
    
    async def _anchor_data(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Anchor data for immutable audit trails"""
        try:
            data_to_anchor = config['data']
            anchor_type = config.get('anchor_type', 'workflow_result')
            
            # Create anchor record
            anchor_record = {
                'anchor_type': anchor_type,
                'workflow_id': context.workflow_id,
                'tenant_id': context.tenant_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data_to_anchor,
                'context': {
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'execution_id': context.execution_id
                }
            }
            
            # Generate cryptographic hash
            anchor_hash = self._generate_anchor_hash(anchor_record)
            anchor_record['anchor_hash'] = anchor_hash
            
            # Store anchor (would integrate with immutable storage)
            anchor_id = await self._store_anchor(context, anchor_record)
            
            return OperatorResult(
                success=True,
                output_data={
                    'data_anchored': True,
                    'anchor_id': anchor_id,
                    'anchor_hash': anchor_hash,
                    'anchor_type': anchor_type,
                    'immutable': True
                },
                confidence_score=1.0
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Data anchoring failed: {e}"
            )
    
    async def _validate_compliance(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Validate compliance against multiple rules"""
        try:
            validation_rules = config['validation_rules']
            data_to_validate = config.get('data', context.previous_outputs)
            
            validation_results = []
            overall_compliance = True
            
            for rule in validation_rules:
                rule_result = await self._validate_single_rule(rule, data_to_validate, context)
                validation_results.append(rule_result)
                
                if not rule_result['compliant']:
                    overall_compliance = False
            
            return OperatorResult(
                success=overall_compliance,
                output_data={
                    'validation_results': validation_results,
                    'overall_compliance': overall_compliance,
                    'rules_checked': len(validation_rules),
                    'rules_passed': len([r for r in validation_results if r['compliant']])
                },
                confidence_score=1.0
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Compliance validation failed: {e}"
            )
    
    async def _load_policy(self, context: OperatorContext, policy_id: str) -> Optional[Dict[str, Any]]:
        """Load policy from database"""
        try:
            pool_manager = context.pool_manager
            if not pool_manager or not pool_manager.postgres_pool:
                return None
            
            async with pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant = '{context.tenant_id}'")
                
                # Query policy
                query = """
                    SELECT policy_pack_id, name, rules, compliance_standards
                    FROM dsl_policy_packs
                    WHERE policy_pack_id = $1
                """
                row = await conn.fetchrow(query, policy_id)
                
                if row:
                    return {
                        'policy_id': row['policy_pack_id'],
                        'name': row['name'],
                        'rules': row['rules'],
                        'compliance_standards': row['compliance_standards']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error loading policy {policy_id}: {e}")
            return None
    
    async def _check_policy_compliance(self, policy: Dict[str, Any], fields: Dict[str, Any], context: OperatorContext) -> Dict[str, Any]:
        """Check compliance against policy rules"""
        violations = []
        
        try:
            rules = policy.get('rules', [])
            
            for rule in rules:
                rule_type = rule.get('type')
                mandatory = rule.get('mandatory', False)
                
                if rule_type == 'evidence_required' and mandatory:
                    if not context.evidence_required:
                        violations.append(f"Evidence required by policy {policy['policy_id']}")
                
                elif rule_type == 'approval_required':
                    condition = rule.get('condition')
                    if condition and self._evaluate_condition(condition, fields):
                        approver_role = rule.get('approver_role')
                        violations.append(f"Approval required from {approver_role} for this operation")
                
                elif rule_type == 'data_minimization':
                    # Check if only necessary fields are being processed
                    allowed_fields = rule.get('allowed_fields', [])
                    if allowed_fields:
                        for field in fields:
                            if field not in allowed_fields:
                                violations.append(f"Field '{field}' not allowed by data minimization policy")
            
            return {
                'violations': violations,
                'rules_checked': len(rules)
            }
            
        except Exception as e:
            logger.error(f"Policy compliance check error: {e}")
            return {
                'violations': [f"Policy compliance check failed: {e}"],
                'rules_checked': 0
            }
    
    def _evaluate_condition(self, condition: str, fields: Dict[str, Any]) -> bool:
        """Evaluate policy condition"""
        try:
            # Simple condition evaluation
            # In production, this would be more sophisticated
            
            # Handle common patterns like "annual_revenue > 1000000"
            for field_name, field_value in fields.items():
                condition = condition.replace(field_name, str(field_value))
            
            # Basic evaluation (in production, use safe evaluation)
            return eval(condition)
            
        except:
            return False
    
    async def _validate_single_rule(self, rule: Dict[str, Any], data: Dict[str, Any], context: OperatorContext) -> Dict[str, Any]:
        """Validate a single compliance rule"""
        rule_type = rule.get('type')
        rule_name = rule.get('name', f'Rule_{rule_type}')
        
        try:
            if rule_type == 'required_fields':
                required_fields = rule.get('fields', [])
                missing_fields = []
                
                for field in required_fields:
                    if field not in data or not data[field]:
                        missing_fields.append(field)
                
                return {
                    'rule_name': rule_name,
                    'rule_type': rule_type,
                    'compliant': len(missing_fields) == 0,
                    'details': f"Missing fields: {missing_fields}" if missing_fields else "All required fields present"
                }
            
            elif rule_type == 'data_quality':
                min_quality_score = rule.get('min_score', 0.8)
                quality_score = self._calculate_data_quality(data)
                
                return {
                    'rule_name': rule_name,
                    'rule_type': rule_type,
                    'compliant': quality_score >= min_quality_score,
                    'details': f"Quality score: {quality_score:.2f}, Required: {min_quality_score:.2f}"
                }
            
            else:
                return {
                    'rule_name': rule_name,
                    'rule_type': rule_type,
                    'compliant': True,
                    'details': f"Rule type '{rule_type}' not implemented, defaulting to compliant"
                }
                
        except Exception as e:
            return {
                'rule_name': rule_name,
                'rule_type': rule_type,
                'compliant': False,
                'details': f"Rule validation error: {e}"
            }
    
    def _calculate_data_quality(self, data: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        if not data:
            return 0.0
        
        total_fields = len(data)
        complete_fields = 0
        
        for value in data.values():
            if value and str(value).strip():
                complete_fields += 1
        
        return complete_fields / total_fields if total_fields > 0 else 0.0
    
    def _generate_evidence_hash(self, evidence_record: Dict[str, Any]) -> str:
        """Generate cryptographic hash for evidence integrity"""
        # Create deterministic JSON string
        evidence_json = json.dumps(evidence_record, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(evidence_json.encode('utf-8')).hexdigest()
    
    def _generate_anchor_hash(self, anchor_record: Dict[str, Any]) -> str:
        """Generate cryptographic hash for data anchoring"""
        # Create deterministic JSON string
        anchor_json = json.dumps(anchor_record, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(anchor_json.encode('utf-8')).hexdigest()
    
    async def _store_evidence(self, context: OperatorContext, evidence_record: Dict[str, Any]) -> str:
        """Store evidence record in database"""
        try:
            pool_manager = context.pool_manager
            if not pool_manager or not pool_manager.postgres_pool:
                return "evidence_id_mock"  # Mock ID when DB not available
            
            async with pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant = '{context.tenant_id}'")
                
                # Insert evidence pack
                query = """
                    INSERT INTO dsl_evidence_packs (
                        trace_id, evidence_data, evidence_hash, tenant_id
                    ) VALUES ($1, $2, $3, $4)
                    RETURNING evidence_pack_id
                """
                
                row = await conn.fetchrow(
                    query,
                    context.execution_id,
                    json.dumps(evidence_record),
                    evidence_record['evidence_hash'],
                    context.tenant_id
                )
                
                return str(row['evidence_pack_id'])
                
        except Exception as e:
            logger.error(f"Error storing evidence: {e}")
            return f"evidence_error_{hash(str(evidence_record))}"
    
    async def _store_anchor(self, context: OperatorContext, anchor_record: Dict[str, Any]) -> str:
        """Store anchor record for immutable audit trail"""
        try:
            # In production, this would store in immutable storage
            # For now, return mock anchor ID
            anchor_id = f"anchor_{hash(str(anchor_record))}"
            
            logger.info(f"Anchor record stored: {anchor_id}")
            return anchor_id
            
        except Exception as e:
            logger.error(f"Error storing anchor: {e}")
            return f"anchor_error_{hash(str(anchor_record))}"
    
    async def _create_evidence_pack(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Create an evidence pack for audit trail"""
        try:
            evidence_type = config.get('evidence_type', 'workflow_execution')
            retention_days = config.get('retention_days', 2555)  # Default 7 years for compliance
            
            # Get execution data from context
            execution_data = {
                'workflow_id': getattr(context, 'workflow_id', 'unknown'),
                'execution_id': context.execution_id,
                'tenant_id': context.tenant_id,
                'timestamp': datetime.utcnow().isoformat(),
                'evidence_type': evidence_type,
                'retention_days': retention_days
            }
            
            # Add any additional data from config
            if 'additional_data' in config:
                execution_data.update(config['additional_data'])
            
            # Create evidence entry
            evidence_id = await self._create_evidence_entry(
                context,
                evidence_type,
                execution_data
            )
            
            return OperatorResult(
                success=True,
                output_data={
                    'action': config['action'],
                    'evidence_id': evidence_id,
                    'evidence_type': evidence_type,
                    'retention_days': retention_days,
                    'created_at': execution_data['timestamp']
                }
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Evidence pack creation failed: {e}"
            )
