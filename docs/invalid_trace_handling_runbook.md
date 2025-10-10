# Invalid Trace Handling Runbook

**Task 7.1-T45: Create runbook for invalid trace handling**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Engineering Team

---

## Overview

This runbook provides comprehensive procedures for handling invalid trace data in the RBA system. It covers detection, classification, remediation, and prevention of invalid traces to maintain data quality and system reliability.

---

## Invalid Trace Classification

### **Severity Levels**

#### **Level 1: Critical (Immediate Action Required)**
- **Schema violations**: Missing required fields, invalid data types
- **Security violations**: PII exposure, tenant isolation breaches
- **Compliance violations**: Missing governance metadata, audit trail gaps
- **Data corruption**: Malformed JSON, encoding issues

#### **Level 2: High (Action Required Within 1 Hour)**
- **Business logic violations**: Invalid workflow states, inconsistent data
- **Performance issues**: Oversized traces, excessive processing time
- **Trust score violations**: Below minimum thresholds
- **Governance warnings**: Policy violations, override anomalies

#### **Level 3: Medium (Action Required Within 4 Hours)**
- **Data quality issues**: Missing optional fields, inconsistent formatting
- **Validation warnings**: Non-critical schema mismatches
- **Performance degradation**: Slower than expected processing
- **Monitoring alerts**: Unusual patterns, statistical anomalies

#### **Level 4: Low (Action Required Within 24 Hours)**
- **Informational issues**: Deprecated field usage, minor inconsistencies
- **Optimization opportunities**: Inefficient data structures
- **Documentation gaps**: Missing metadata, unclear field values

---

## Detection and Monitoring

### **Automated Detection System**

```python
class InvalidTraceDetector:
    """Detects and classifies invalid traces"""
    
    def __init__(self):
        self.validators = [
            SchemaValidator(),
            SecurityValidator(),
            ComplianceValidator(),
            BusinessLogicValidator(),
            PerformanceValidator()
        ]
        self.classification_rules = self._load_classification_rules()
    
    async def detect_invalid_traces(self, traces: List[Dict[str, Any]]) -> List[InvalidTraceReport]:
        """Detect invalid traces and classify issues"""
        
        invalid_reports = []
        
        for trace_data in traces:
            issues = []
            
            # Run all validators
            for validator in self.validators:
                validation_result = await validator.validate(trace_data)
                if not validation_result.valid:
                    issues.extend(validation_result.issues)
            
            if issues:
                # Classify severity
                severity = self._classify_severity(issues)
                
                # Create report
                report = InvalidTraceReport(
                    trace_id=trace_data.get('trace_id', 'unknown'),
                    tenant_id=trace_data.get('context', {}).get('tenant_id'),
                    severity=severity,
                    issues=issues,
                    detected_at=datetime.now(timezone.utc),
                    raw_data=trace_data
                )
                
                invalid_reports.append(report)
        
        return invalid_reports
    
    def _classify_severity(self, issues: List[ValidationIssue]) -> str:
        """Classify overall severity based on individual issues"""
        
        severity_scores = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        
        max_severity = 0
        for issue in issues:
            score = severity_scores.get(issue.severity, 0)
            max_severity = max(max_severity, score)
        
        severity_map = {4: 'critical', 3: 'high', 2: 'medium', 1: 'low'}
        return severity_map.get(max_severity, 'low')

@dataclass
class InvalidTraceReport:
    """Report for invalid trace detection"""
    
    trace_id: str
    tenant_id: Optional[int]
    severity: str
    issues: List[ValidationIssue]
    detected_at: datetime
    raw_data: Dict[str, Any]
    
    # Processing status
    status: str = "detected"  # detected, triaged, in_progress, resolved, ignored
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Escalation
    escalated: bool = False
    escalated_at: Optional[datetime] = None
    escalation_reason: Optional[str] = None
```

### **Real-Time Monitoring**

```python
class TraceQualityMonitor:
    """Real-time monitoring of trace quality"""
    
    def __init__(self):
        self.metrics = TraceQualityMetrics()
        self.alert_manager = AlertManager()
        self.thresholds = self._load_quality_thresholds()
    
    async def monitor_trace_quality(self):
        """Continuous monitoring of trace quality"""
        
        while True:
            try:
                # Get recent traces
                recent_traces = await self._get_recent_traces(minutes=5)
                
                # Calculate quality metrics
                quality_metrics = await self._calculate_quality_metrics(recent_traces)
                
                # Check thresholds and alert
                await self._check_thresholds_and_alert(quality_metrics)
                
                # Update dashboards
                await self._update_quality_dashboards(quality_metrics)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in trace quality monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_quality_metrics(self, traces: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trace quality metrics"""
        
        if not traces:
            return {}
        
        total_traces = len(traces)
        invalid_traces = 0
        critical_issues = 0
        schema_violations = 0
        security_violations = 0
        
        for trace in traces:
            validation_result = await self._validate_trace(trace)
            
            if not validation_result.valid:
                invalid_traces += 1
                
                for issue in validation_result.issues:
                    if issue.severity == 'critical':
                        critical_issues += 1
                    if issue.category == 'schema':
                        schema_violations += 1
                    if issue.category == 'security':
                        security_violations += 1
        
        return {
            'invalid_trace_rate': invalid_traces / total_traces,
            'critical_issue_rate': critical_issues / total_traces,
            'schema_violation_rate': schema_violations / total_traces,
            'security_violation_rate': security_violations / total_traces,
            'total_traces_processed': total_traces
        }
```

---

## Response Procedures

### **Critical Issues (Level 1) - Immediate Response**

```markdown
# CRITICAL INVALID TRACE RESPONSE

## Immediate Actions (0-5 minutes)
1. **Alert Acknowledgment**
   - Acknowledge alert in monitoring system
   - Notify on-call engineer via PagerDuty
   - Create incident ticket

2. **Impact Assessment**
   - Check affected tenant(s)
   - Assess data corruption scope
   - Verify system stability

3. **Immediate Containment**
   - Stop processing affected traces
   - Quarantine invalid data
   - Enable circuit breaker if needed

## Investigation (5-15 minutes)
1. **Root Cause Analysis**
   ```bash
   # Check recent deployments
   kubectl get deployments -o wide
   
   # Review trace processing logs
   kubectl logs -l app=trace-processor --since=1h
   
   # Check schema registry status
   curl -s $SCHEMA_REGISTRY_URL/health
   ```

2. **Data Analysis**
   ```bash
   # Query invalid traces
   python investigate_invalid_traces.py --severity=critical --since=1h
   
   # Check trace patterns
   python analyze_trace_patterns.py --trace-ids=<TRACE_IDS>
   ```

## Resolution (15-60 minutes)
1. **Fix Root Cause**
   - Deploy hotfix if code issue
   - Update schema if schema issue
   - Restart services if infrastructure issue

2. **Data Recovery**
   - Restore from backup if needed
   - Reprocess valid traces
   - Update affected records

3. **Validation**
   - Verify fix effectiveness
   - Test trace processing
   - Monitor for recurrence
```

### **High Priority Issues (Level 2) - 1 Hour Response**

```python
class HighPriorityTraceHandler:
    """Handles high priority invalid trace issues"""
    
    async def handle_high_priority_issue(self, report: InvalidTraceReport):
        """Handle high priority invalid trace"""
        
        # Create incident
        incident = await self._create_incident(report)
        
        # Assign to appropriate team
        team = self._determine_responsible_team(report.issues)
        await self._assign_incident(incident.id, team)
        
        # Immediate actions
        await self._quarantine_trace(report.trace_id)
        await self._notify_stakeholders(report, incident.id)
        
        # Start investigation
        investigation_result = await self._investigate_trace_issue(report)
        
        # Apply remediation
        remediation_plan = await self._create_remediation_plan(investigation_result)
        await self._execute_remediation(remediation_plan)
        
        # Verify resolution
        verification_result = await self._verify_resolution(report.trace_id)
        
        if verification_result.resolved:
            await self._close_incident(incident.id, verification_result.notes)
        else:
            await self._escalate_incident(incident.id, "Resolution verification failed")
    
    def _determine_responsible_team(self, issues: List[ValidationIssue]) -> str:
        """Determine which team should handle the issues"""
        
        team_mapping = {
            'schema': 'platform_engineering',
            'security': 'security_team',
            'compliance': 'compliance_team',
            'business_logic': 'product_engineering',
            'performance': 'platform_engineering'
        }
        
        # Find the most critical issue category
        critical_categories = [issue.category for issue in issues if issue.severity == 'critical']
        
        if critical_categories:
            return team_mapping.get(critical_categories[0], 'platform_engineering')
        
        # Default to platform engineering
        return 'platform_engineering'
```

---

## Remediation Strategies

### **Automatic Remediation**

```python
class AutomaticRemediationEngine:
    """Automatically remediates common invalid trace issues"""
    
    def __init__(self):
        self.remediation_rules = self._load_remediation_rules()
        self.success_rate_threshold = 0.95
    
    async def attempt_automatic_remediation(self, report: InvalidTraceReport) -> RemediationResult:
        """Attempt automatic remediation of invalid trace"""
        
        remediation_result = RemediationResult(
            trace_id=report.trace_id,
            attempted=True,
            successful=False,
            actions_taken=[],
            manual_intervention_required=False
        )
        
        for issue in report.issues:
            if issue.category in self.remediation_rules:
                rule = self.remediation_rules[issue.category]
                
                if rule.auto_remediable and rule.success_rate >= self.success_rate_threshold:
                    try:
                        action_result = await self._apply_remediation_rule(report.raw_data, rule)
                        remediation_result.actions_taken.append(action_result)
                        
                        if action_result.successful:
                            remediation_result.successful = True
                        
                    except Exception as e:
                        remediation_result.actions_taken.append({
                            'rule': rule.name,
                            'successful': False,
                            'error': str(e)
                        })
                else:
                    remediation_result.manual_intervention_required = True
        
        return remediation_result
    
    async def _apply_remediation_rule(self, trace_data: Dict[str, Any], rule: RemediationRule) -> Dict[str, Any]:
        """Apply specific remediation rule"""
        
        if rule.name == "missing_required_fields":
            return await self._fix_missing_required_fields(trace_data, rule)
        elif rule.name == "invalid_data_types":
            return await self._fix_invalid_data_types(trace_data, rule)
        elif rule.name == "schema_version_mismatch":
            return await self._fix_schema_version_mismatch(trace_data, rule)
        elif rule.name == "tenant_isolation_violation":
            return await self._fix_tenant_isolation_violation(trace_data, rule)
        else:
            raise UnsupportedRemediationRuleError(f"Unknown rule: {rule.name}")
    
    async def _fix_missing_required_fields(self, trace_data: Dict[str, Any], rule: RemediationRule) -> Dict[str, Any]:
        """Fix missing required fields"""
        
        fixed_data = trace_data.copy()
        actions = []
        
        for field_path, default_value in rule.parameters.get('field_defaults', {}).items():
            if not self._has_nested_field(fixed_data, field_path):
                self._set_nested_field(fixed_data, field_path, default_value)
                actions.append(f"Added missing field {field_path} with default value")
        
        # Validate fix
        validator = SchemaValidator()
        validation_result = await validator.validate(fixed_data)
        
        if validation_result.valid:
            # Store fixed trace
            await self._store_fixed_trace(fixed_data)
            
            return {
                'rule': rule.name,
                'successful': True,
                'actions': actions,
                'fixed_trace_id': fixed_data.get('trace_id')
            }
        else:
            return {
                'rule': rule.name,
                'successful': False,
                'actions': actions,
                'validation_errors': validation_result.errors
            }
```

### **Manual Remediation Workflows**

```python
class ManualRemediationWorkflow:
    """Manages manual remediation workflows"""
    
    async def create_manual_remediation_task(self, report: InvalidTraceReport) -> RemediationTask:
        """Create manual remediation task"""
        
        task = RemediationTask(
            task_id=str(uuid.uuid4()),
            trace_id=report.trace_id,
            tenant_id=report.tenant_id,
            severity=report.severity,
            issues=report.issues,
            created_at=datetime.now(timezone.utc),
            status="pending",
            assigned_to=None
        )
        
        # Determine task type and priority
        task.task_type = self._determine_task_type(report.issues)
        task.priority = self._calculate_task_priority(report.severity, report.tenant_id)
        
        # Create detailed instructions
        task.instructions = await self._generate_remediation_instructions(report)
        
        # Assign to appropriate engineer
        task.assigned_to = await self._assign_remediation_task(task)
        
        # Store task
        await self._store_remediation_task(task)
        
        # Notify assignee
        await self._notify_task_assignee(task)
        
        return task
    
    async def _generate_remediation_instructions(self, report: InvalidTraceReport) -> str:
        """Generate detailed remediation instructions"""
        
        instructions = [
            f"# Manual Remediation Required",
            f"**Trace ID:** {report.trace_id}",
            f"**Tenant ID:** {report.tenant_id}",
            f"**Severity:** {report.severity}",
            f"**Detected:** {report.detected_at.isoformat()}",
            "",
            "## Issues Identified:"
        ]
        
        for i, issue in enumerate(report.issues, 1):
            instructions.extend([
                f"{i}. **{issue.category.title()} Issue** (Severity: {issue.severity})",
                f"   - Description: {issue.description}",
                f"   - Field: {issue.field_path}",
                f"   - Expected: {issue.expected_value}",
                f"   - Actual: {issue.actual_value}",
                ""
            ])
        
        instructions.extend([
            "## Remediation Steps:",
            "1. Review the trace data in the investigation dashboard",
            "2. Identify the root cause of each issue",
            "3. Apply appropriate fixes using the remediation tools",
            "4. Validate the fixed trace against the current schema",
            "5. Update the remediation task with resolution notes",
            "",
            "## Tools and Resources:",
            "- Trace Investigation Dashboard: https://dashboard.company.com/traces/investigate",
            "- Schema Validation Tool: `python validate_trace.py --trace-id={trace_id}`",
            "- Trace Editor: `python edit_trace.py --trace-id={trace_id}`",
            "- Documentation: https://docs.company.com/trace-remediation"
        ])
        
        return "\n".join(instructions)
```

---

## Prevention Strategies

### **Proactive Quality Gates**

```python
class TraceQualityGates:
    """Implements quality gates to prevent invalid traces"""
    
    def __init__(self):
        self.pre_ingestion_validators = [
            SchemaPreValidator(),
            SecurityPreValidator(),
            BusinessLogicPreValidator()
        ]
        self.quality_thresholds = self._load_quality_thresholds()
    
    async def validate_before_ingestion(self, trace_data: Dict[str, Any]) -> QualityGateResult:
        """Validate trace before ingestion"""
        
        result = QualityGateResult(
            passed=True,
            issues=[],
            recommendations=[]
        )
        
        # Run pre-ingestion validators
        for validator in self.pre_ingestion_validators:
            validation_result = await validator.validate(trace_data)
            
            if not validation_result.valid:
                result.issues.extend(validation_result.issues)
                
                # Check if issues are blocking
                blocking_issues = [issue for issue in validation_result.issues if issue.blocking]
                if blocking_issues:
                    result.passed = False
        
        # Generate recommendations
        if result.issues:
            result.recommendations = await self._generate_quality_recommendations(result.issues)
        
        return result
    
    async def _generate_quality_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate recommendations to improve trace quality"""
        
        recommendations = []
        
        issue_categories = set(issue.category for issue in issues)
        
        if 'schema' in issue_categories:
            recommendations.append(
                "Consider updating your trace generation code to match the latest schema version"
            )
        
        if 'security' in issue_categories:
            recommendations.append(
                "Review PII handling and ensure proper redaction is applied"
            )
        
        if 'performance' in issue_categories:
            recommendations.append(
                "Optimize trace data size and structure for better performance"
            )
        
        return recommendations
```

### **Developer Education and Tooling**

```python
class TraceQualityEducation:
    """Provides education and tooling for trace quality"""
    
    def generate_quality_report_for_tenant(self, tenant_id: int) -> TenantQualityReport:
        """Generate quality report for tenant"""
        
        # Get tenant's trace quality metrics
        metrics = self._get_tenant_quality_metrics(tenant_id)
        
        # Identify common issues
        common_issues = self._identify_common_issues(tenant_id)
        
        # Generate recommendations
        recommendations = self._generate_tenant_recommendations(metrics, common_issues)
        
        # Create educational content
        educational_content = self._generate_educational_content(common_issues)
        
        return TenantQualityReport(
            tenant_id=tenant_id,
            quality_score=metrics.overall_score,
            common_issues=common_issues,
            recommendations=recommendations,
            educational_content=educational_content,
            improvement_plan=self._create_improvement_plan(tenant_id, metrics)
        )
    
    def create_trace_quality_sdk(self) -> TraceQualitySDK:
        """Create SDK for trace quality validation"""
        
        return TraceQualitySDK(
            validators=self.pre_ingestion_validators,
            schema_registry=self.schema_registry,
            quality_guidelines=self._load_quality_guidelines(),
            best_practices=self._load_best_practices()
        )
```

---

## Escalation Procedures

### **Escalation Matrix**

```yaml
escalation_matrix:
  critical_issues:
    immediate: ["on_call_engineer", "platform_lead"]
    15_minutes: ["engineering_manager", "cto"]
    30_minutes: ["ceo", "board_notification"]
    
  high_priority_issues:
    1_hour: ["assigned_engineer", "team_lead"]
    4_hours: ["engineering_manager"]
    24_hours: ["platform_lead"]
    
  compliance_violations:
    immediate: ["compliance_officer", "legal_team"]
    30_minutes: ["cto", "ceo"]
    
  security_violations:
    immediate: ["security_team", "ciso"]
    15_minutes: ["cto", "incident_commander"]
```

### **Automated Escalation**

```python
class EscalationManager:
    """Manages automatic escalation of invalid trace issues"""
    
    async def check_escalation_triggers(self):
        """Check for escalation triggers"""
        
        # Get unresolved issues
        unresolved_issues = await self._get_unresolved_issues()
        
        for issue in unresolved_issues:
            escalation_needed = await self._check_escalation_criteria(issue)
            
            if escalation_needed:
                await self._escalate_issue(issue, escalation_needed.reason)
    
    async def _check_escalation_criteria(self, issue: InvalidTraceReport) -> Optional[EscalationTrigger]:
        """Check if issue meets escalation criteria"""
        
        now = datetime.now(timezone.utc)
        age_minutes = (now - issue.detected_at).total_seconds() / 60
        
        # Time-based escalation
        if issue.severity == 'critical' and age_minutes > 15:
            return EscalationTrigger(
                reason="Critical issue unresolved for >15 minutes",
                escalation_level="executive"
            )
        
        if issue.severity == 'high' and age_minutes > 60:
            return EscalationTrigger(
                reason="High priority issue unresolved for >1 hour",
                escalation_level="management"
            )
        
        # Volume-based escalation
        similar_issues = await self._count_similar_issues(issue, hours=1)
        if similar_issues > 10:
            return EscalationTrigger(
                reason=f"High volume of similar issues: {similar_issues}",
                escalation_level="management"
            )
        
        return None
```

---

## Operational Dashboards

### **Invalid Trace Dashboard Configuration**

```yaml
dashboard_config:
  name: "Invalid Trace Monitoring"
  refresh_interval: "30s"
  
  panels:
    - title: "Invalid Trace Rate"
      type: "stat"
      query: "rate(invalid_traces_total[5m])"
      thresholds: [0.01, 0.05, 0.1]
      
    - title: "Issues by Severity"
      type: "pie_chart"
      query: "sum by (severity) (invalid_trace_issues)"
      
    - title: "Issues by Category"
      type: "bar_chart"
      query: "sum by (category) (invalid_trace_issues)"
      
    - title: "Resolution Time"
      type: "histogram"
      query: "histogram_quantile(0.95, rate(issue_resolution_time_bucket[5m]))"
      
    - title: "Top Affected Tenants"
      type: "table"
      query: "topk(10, sum by (tenant_id) (invalid_traces_total))"
      
    - title: "Recent Critical Issues"
      type: "logs"
      query: "severity=critical AND status!=resolved"
      limit: 50

alerts:
  - name: "HighInvalidTraceRate"
    condition: "rate(invalid_traces_total[5m]) > 0.05"
    severity: "warning"
    
  - name: "CriticalTraceIssue"
    condition: "increase(invalid_trace_issues{severity='critical'}[1m]) > 0"
    severity: "critical"
    
  - name: "UnresolvedCriticalIssues"
    condition: "count(invalid_trace_issues{severity='critical',status!='resolved'}) > 0"
    severity: "critical"
```

---

## Testing and Validation

### **Runbook Testing**

```python
class RunbookTester:
    """Tests invalid trace handling procedures"""
    
    async def test_critical_issue_response(self):
        """Test critical issue response procedures"""
        
        # Create synthetic critical issue
        synthetic_trace = self._create_invalid_trace(severity='critical')
        
        # Inject into system
        await self._inject_invalid_trace(synthetic_trace)
        
        # Verify detection
        detection_time = await self._measure_detection_time()
        assert detection_time < 30, "Detection took too long"
        
        # Verify alert
        alert_sent = await self._verify_alert_sent()
        assert alert_sent, "Alert not sent"
        
        # Verify containment
        containment_applied = await self._verify_containment()
        assert containment_applied, "Containment not applied"
        
        # Cleanup
        await self._cleanup_test_data()
    
    async def test_remediation_procedures(self):
        """Test remediation procedures"""
        
        # Test automatic remediation
        auto_remediation_result = await self._test_automatic_remediation()
        assert auto_remediation_result.success_rate > 0.9
        
        # Test manual remediation workflow
        manual_workflow_result = await self._test_manual_workflow()
        assert manual_workflow_result.completed_successfully
        
        # Test escalation procedures
        escalation_result = await self._test_escalation_procedures()
        assert escalation_result.escalated_correctly
```

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** platform-engineering@company.com
