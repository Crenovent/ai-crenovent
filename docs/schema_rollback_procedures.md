# Schema Rollback Procedures for RBA Traces

**Task 7.1-T35: Define rollback path for schema changes**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Engineering Team

---

## Overview

This document defines comprehensive rollback procedures for RBA trace schema changes to ensure operational safety and zero-downtime deployments. All schema changes must have a defined rollback path before implementation.

---

## Rollback Strategy Framework

### **Schema Change Categories**

#### **Category 1: Additive Changes (Low Risk)**
- Adding new optional fields
- Adding new enum values
- Adding new validation rules (non-breaking)
- **Rollback Complexity**: Simple
- **Rollback Time**: < 5 minutes

#### **Category 2: Modification Changes (Medium Risk)**
- Changing field types (compatible)
- Modifying validation rules
- Updating default values
- **Rollback Complexity**: Moderate
- **Rollback Time**: 5-15 minutes

#### **Category 3: Breaking Changes (High Risk)**
- Removing fields
- Changing field types (incompatible)
- Renaming fields
- **Rollback Complexity**: Complex
- **Rollback Time**: 15-60 minutes

#### **Category 4: Structural Changes (Critical Risk)**
- Changing schema version format
- Modifying core trace structure
- Altering tenant isolation model
- **Rollback Complexity**: Very Complex
- **Rollback Time**: 1-4 hours

---

## Dynamic Rollback Configuration

### **Schema Version Management**

```python
@dataclass
class SchemaVersion:
    """Schema version with rollback metadata"""
    
    version: str  # e.g., "1.2.3"
    release_date: str
    change_category: str  # additive, modification, breaking, structural
    
    # Rollback information
    rollback_supported: bool = True
    rollback_to_version: Optional[str] = None
    rollback_complexity: str = "simple"  # simple, moderate, complex, very_complex
    rollback_time_estimate_minutes: int = 5
    
    # Compatibility
    backward_compatible: bool = True
    forward_compatible: bool = False
    
    # Dependencies
    dependent_services: List[str] = field(default_factory=list)
    database_migration_required: bool = False
    
    # Validation
    rollback_tested: bool = False
    rollback_test_date: Optional[str] = None
    
    # Emergency procedures
    emergency_rollback_available: bool = True
    emergency_contact: str = "platform-engineering@company.com"

class SchemaRollbackManager:
    """Manages schema rollback operations"""
    
    def __init__(self):
        self.version_history = self._load_version_history()
        self.rollback_procedures = self._load_rollback_procedures()
        self.current_version = self._get_current_version()
    
    def can_rollback_to(self, target_version: str) -> Dict[str, Any]:
        """Check if rollback to target version is possible"""
        
        current = self.version_history[self.current_version]
        target = self.version_history.get(target_version)
        
        if not target:
            return {
                'possible': False,
                'reason': f'Target version {target_version} not found'
            }
        
        # Check rollback path
        rollback_path = self._find_rollback_path(self.current_version, target_version)
        
        if not rollback_path:
            return {
                'possible': False,
                'reason': 'No valid rollback path found'
            }
        
        # Calculate complexity and time
        total_complexity = self._calculate_rollback_complexity(rollback_path)
        total_time = self._calculate_rollback_time(rollback_path)
        
        return {
            'possible': True,
            'rollback_path': rollback_path,
            'complexity': total_complexity,
            'estimated_time_minutes': total_time,
            'requires_downtime': total_complexity in ['complex', 'very_complex'],
            'affected_services': self._get_affected_services(rollback_path)
        }
```

### **Rollback Execution Engine**

```python
class RollbackExecutor:
    """Executes schema rollback operations"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.rollback_log = []
        self.validation_enabled = True
    
    async def execute_rollback(
        self, 
        target_version: str, 
        rollback_reason: str,
        approved_by: str
    ) -> Dict[str, Any]:
        """Execute schema rollback to target version"""
        
        rollback_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            # Pre-rollback validation
            validation_result = await self._pre_rollback_validation(target_version)
            if not validation_result['valid']:
                raise RollbackValidationError(validation_result['errors'])
            
            # Create rollback plan
            rollback_plan = await self._create_rollback_plan(target_version)
            
            # Execute rollback steps
            for step in rollback_plan['steps']:
                step_result = await self._execute_rollback_step(step)
                self.rollback_log.append({
                    'step': step['name'],
                    'status': step_result['status'],
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'details': step_result.get('details', {})
                })
                
                if step_result['status'] == 'failed':
                    raise RollbackExecutionError(f"Step {step['name']} failed: {step_result['error']}")
            
            # Post-rollback validation
            post_validation = await self._post_rollback_validation(target_version)
            if not post_validation['valid']:
                # Attempt forward recovery
                await self._attempt_forward_recovery()
                raise RollbackValidationError("Post-rollback validation failed")
            
            # Update current version
            await self._update_current_version(target_version)
            
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                'rollback_id': rollback_id,
                'status': 'success',
                'target_version': target_version,
                'execution_time_seconds': execution_time,
                'steps_executed': len(rollback_plan['steps']),
                'rollback_log': self.rollback_log
            }
        
        except Exception as e:
            # Emergency recovery procedures
            await self._emergency_recovery(rollback_id, str(e))
            raise
    
    async def _create_rollback_plan(self, target_version: str) -> Dict[str, Any]:
        """Create detailed rollback execution plan"""
        
        rollback_path = self._find_rollback_path(self.current_version, target_version)
        steps = []
        
        for version_step in rollback_path:
            current_ver = version_step['from']
            target_ver = version_step['to']
            
            # Database rollback steps
            if version_step['database_changes']:
                steps.append({
                    'name': f'rollback_database_{current_ver}_to_{target_ver}',
                    'type': 'database',
                    'action': 'execute_migration',
                    'migration_file': f'rollback_{current_ver}_to_{target_ver}.sql',
                    'timeout_seconds': 300
                })
            
            # Schema registry rollback
            steps.append({
                'name': f'update_schema_registry_{target_ver}',
                'type': 'schema_registry',
                'action': 'update_version',
                'target_version': target_ver,
                'timeout_seconds': 30
            })
            
            # Service restart (if required)
            if version_step['requires_restart']:
                steps.append({
                    'name': f'restart_services_{target_ver}',
                    'type': 'service_management',
                    'action': 'rolling_restart',
                    'services': version_step['affected_services'],
                    'timeout_seconds': 600
                })
            
            # Validation step
            steps.append({
                'name': f'validate_rollback_{target_ver}',
                'type': 'validation',
                'action': 'run_validation_suite',
                'target_version': target_ver,
                'timeout_seconds': 120
            })
        
        return {
            'rollback_id': str(uuid.uuid4()),
            'target_version': target_version,
            'steps': steps,
            'estimated_time_minutes': sum(step.get('timeout_seconds', 0) for step in steps) / 60
        }
```

---

## Rollback Procedures by Change Type

### **1. Additive Changes Rollback**

```yaml
# Example: Rolling back addition of new optional field
additive_rollback_procedure:
  change_type: "additive"
  example: "Added optional 'risk_assessment' field to GovernanceEvent"
  
  rollback_steps:
    1. "Update schema registry to previous version"
    2. "Deploy service with previous schema version"
    3. "Validate existing traces still process correctly"
    4. "Monitor for any compatibility issues"
  
  rollback_script: |
    # Update schema version
    kubectl patch configmap trace-schema-config \
      --patch '{"data":{"schema_version":"1.2.2"}}'
    
    # Rolling restart of trace processors
    kubectl rollout restart deployment/trace-processor
    
    # Validate rollback
    python validate_schema_rollback.py --target-version=1.2.2
  
  validation_checks:
    - "Schema registry shows correct version"
    - "New traces validate against old schema"
    - "Existing traces remain accessible"
    - "No service errors in logs"
  
  rollback_time_estimate: "5 minutes"
  downtime_required: false
```

### **2. Breaking Changes Rollback**

```yaml
# Example: Rolling back field removal
breaking_rollback_procedure:
  change_type: "breaking"
  example: "Removed deprecated 'legacy_trust_score' field"
  
  rollback_steps:
    1. "Stop trace ingestion temporarily"
    2. "Restore database schema with removed field"
    3. "Backfill missing field data from backup"
    4. "Update schema registry"
    5. "Deploy previous service version"
    6. "Resume trace ingestion"
    7. "Validate data integrity"
  
  rollback_script: |
    # Stop ingestion
    kubectl scale deployment/trace-ingester --replicas=0
    
    # Restore database schema
    psql -h $DB_HOST -d $DB_NAME -f rollback_v1.3.0_to_v1.2.5.sql
    
    # Backfill data
    python backfill_legacy_trust_score.py --from-backup
    
    # Update schema registry
    curl -X PUT $SCHEMA_REGISTRY/schemas/trace-schema/versions/1.2.5
    
    # Deploy previous version
    kubectl set image deployment/trace-processor \
      trace-processor=trace-processor:v1.2.5
    
    # Resume ingestion
    kubectl scale deployment/trace-ingester --replicas=3
    
    # Validate
    python validate_breaking_rollback.py --target-version=1.2.5
  
  validation_checks:
    - "Database schema matches target version"
    - "All required fields present and populated"
    - "Services running target version"
    - "Trace ingestion resumed successfully"
    - "No data loss detected"
  
  rollback_time_estimate: "30-45 minutes"
  downtime_required: true
  downtime_estimate: "10-15 minutes"
```

### **3. Emergency Rollback Procedures**

```python
class EmergencyRollbackProcedures:
    """Emergency rollback procedures for critical issues"""
    
    @staticmethod
    async def emergency_rollback_to_last_known_good():
        """Emergency rollback to last known good version"""
        
        # Get last known good version
        last_good_version = await get_last_known_good_version()
        
        # Immediate service rollback
        await execute_immediate_service_rollback(last_good_version)
        
        # Database rollback (if safe)
        if await is_database_rollback_safe():
            await execute_database_rollback(last_good_version)
        
        # Notify stakeholders
        await send_emergency_notification(
            "Emergency schema rollback executed",
            f"Rolled back to version {last_good_version}"
        )
    
    @staticmethod
    async def circuit_breaker_rollback():
        """Automatic rollback triggered by circuit breaker"""
        
        # Stop all trace processing
        await stop_trace_processing()
        
        # Switch to read-only mode
        await enable_read_only_mode()
        
        # Rollback to safe version
        await emergency_rollback_to_last_known_good()
        
        # Gradual service restoration
        await gradual_service_restoration()
```

---

## Validation and Testing

### **Pre-Rollback Validation**

```python
class PreRollbackValidator:
    """Validates system state before rollback"""
    
    async def validate_rollback_readiness(self, target_version: str) -> Dict[str, Any]:
        """Comprehensive pre-rollback validation"""
        
        validation_results = {
            'ready': True,
            'checks': {},
            'warnings': [],
            'blockers': []
        }
        
        # Check 1: Data compatibility
        data_check = await self._validate_data_compatibility(target_version)
        validation_results['checks']['data_compatibility'] = data_check
        if not data_check['passed']:
            validation_results['blockers'].extend(data_check['issues'])
            validation_results['ready'] = False
        
        # Check 2: Service dependencies
        dependency_check = await self._validate_service_dependencies(target_version)
        validation_results['checks']['service_dependencies'] = dependency_check
        if not dependency_check['passed']:
            validation_results['warnings'].extend(dependency_check['issues'])
        
        # Check 3: Backup availability
        backup_check = await self._validate_backup_availability()
        validation_results['checks']['backup_availability'] = backup_check
        if not backup_check['passed']:
            validation_results['blockers'].extend(backup_check['issues'])
            validation_results['ready'] = False
        
        # Check 4: Resource availability
        resource_check = await self._validate_resource_availability()
        validation_results['checks']['resource_availability'] = resource_check
        
        return validation_results
```

### **Post-Rollback Validation**

```python
class PostRollbackValidator:
    """Validates system state after rollback"""
    
    async def validate_rollback_success(self, target_version: str) -> Dict[str, Any]:
        """Comprehensive post-rollback validation"""
        
        validation_results = {
            'successful': True,
            'checks': {},
            'issues': []
        }
        
        # Check 1: Schema version consistency
        version_check = await self._validate_schema_version_consistency(target_version)
        validation_results['checks']['version_consistency'] = version_check
        
        # Check 2: Data integrity
        integrity_check = await self._validate_data_integrity()
        validation_results['checks']['data_integrity'] = integrity_check
        
        # Check 3: Service health
        health_check = await self._validate_service_health()
        validation_results['checks']['service_health'] = health_check
        
        # Check 4: Functional tests
        functional_check = await self._run_functional_tests()
        validation_results['checks']['functional_tests'] = functional_check
        
        # Determine overall success
        for check_name, check_result in validation_results['checks'].items():
            if not check_result['passed']:
                validation_results['successful'] = False
                validation_results['issues'].extend(check_result['issues'])
        
        return validation_results
```

---

## Monitoring and Alerting

### **Rollback Monitoring Dashboard**

```yaml
rollback_monitoring_metrics:
  - name: "schema_rollback_in_progress"
    type: "gauge"
    description: "Whether a schema rollback is currently in progress"
    
  - name: "rollback_execution_time"
    type: "histogram"
    description: "Time taken to execute schema rollbacks"
    
  - name: "rollback_success_rate"
    type: "counter"
    description: "Success rate of schema rollbacks"
    
  - name: "post_rollback_validation_failures"
    type: "counter"
    description: "Number of post-rollback validation failures"

rollback_alerts:
  - name: "RollbackInProgress"
    condition: "schema_rollback_in_progress == 1"
    severity: "warning"
    message: "Schema rollback in progress"
    
  - name: "RollbackFailed"
    condition: "increase(rollback_success_rate{status='failed'}[5m]) > 0"
    severity: "critical"
    message: "Schema rollback failed"
    
  - name: "RollbackTakingTooLong"
    condition: "schema_rollback_in_progress == 1 and time() - rollback_start_time > 3600"
    severity: "critical"
    message: "Schema rollback taking longer than 1 hour"
```

---

## Operational Runbooks

### **Standard Rollback Runbook**

```markdown
# Standard Schema Rollback Procedure

## Prerequisites
- [ ] Rollback approved by Platform Engineering Lead
- [ ] Backup verified and accessible
- [ ] Maintenance window scheduled (if required)
- [ ] Stakeholders notified

## Execution Steps

### Step 1: Pre-Rollback Validation
```bash
python validate_rollback_readiness.py --target-version=<VERSION>
```

### Step 2: Create Rollback Plan
```bash
python create_rollback_plan.py --target-version=<VERSION> --output=rollback_plan.json
```

### Step 3: Execute Rollback
```bash
python execute_rollback.py --plan=rollback_plan.json --approved-by=<USER>
```

### Step 4: Post-Rollback Validation
```bash
python validate_rollback_success.py --target-version=<VERSION>
```

### Step 5: Update Documentation
- [ ] Update schema version in documentation
- [ ] Record rollback in change log
- [ ] Update monitoring dashboards

## Rollback Checklist
- [ ] All services running target version
- [ ] Database schema matches target version
- [ ] Functional tests passing
- [ ] No critical errors in logs
- [ ] Monitoring shows healthy metrics
```

### **Emergency Rollback Runbook**

```markdown
# Emergency Schema Rollback Procedure

## Immediate Actions (< 5 minutes)
1. **Stop Traffic**: Scale ingestion services to 0 replicas
2. **Enable Circuit Breaker**: Activate emergency circuit breaker
3. **Notify Stakeholders**: Send emergency notification

## Emergency Rollback (5-15 minutes)
1. **Identify Last Good Version**: Check monitoring for last stable version
2. **Execute Emergency Rollback**: Run emergency rollback script
3. **Validate Critical Functions**: Ensure basic functionality works

## Recovery and Stabilization (15-60 minutes)
1. **Full Validation**: Run complete validation suite
2. **Gradual Traffic Restoration**: Slowly restore traffic
3. **Monitor Closely**: Watch all metrics and logs
4. **Document Incident**: Record what happened and why

## Post-Incident Actions
1. **Root Cause Analysis**: Investigate what caused the need for emergency rollback
2. **Improve Procedures**: Update rollback procedures based on lessons learned
3. **Test Improvements**: Validate improved procedures in staging
```

---

## Configuration Examples

### **Tenant-Specific Rollback Rules**

```yaml
# High-value tenant with strict SLA
tenant_1000_rollback_config:
  max_rollback_time_minutes: 15
  requires_approval: true
  approval_required_from: ["platform_lead", "tenant_success_manager"]
  notification_channels: ["slack", "email", "pagerduty"]
  validation_level: "comprehensive"
  
# Development tenant with relaxed requirements  
tenant_9999_rollback_config:
  max_rollback_time_minutes: 60
  requires_approval: false
  notification_channels: ["slack"]
  validation_level: "basic"
```

### **Industry-Specific Considerations**

```yaml
# Banking industry - strict compliance requirements
banking_rollback_config:
  regulatory_approval_required: true
  audit_trail_mandatory: true
  data_retention_during_rollback: true
  compliance_validation_required: true
  
# SaaS industry - focus on uptime
saas_rollback_config:
  zero_downtime_preferred: true
  canary_rollback_supported: true
  automated_rollback_triggers: ["error_rate > 5%", "latency > 2s"]
```

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** platform-engineering@company.com
