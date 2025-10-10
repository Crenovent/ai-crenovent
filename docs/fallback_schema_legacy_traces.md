# Fallback Schema for Legacy Traces

**Task 7.1-T38: Document fallback schema for legacy traces**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Engineering Team

---

## Overview

This document defines the fallback schema mechanism for handling legacy trace data that doesn't conform to the current canonical trace schema. The fallback system ensures backward compatibility while maintaining data integrity and governance requirements.

---

## Fallback Schema Architecture

### **Schema Evolution Timeline**

```yaml
schema_versions:
  v0.9.0:
    status: "legacy"
    support_level: "fallback_only"
    deprecation_date: "2024-01-01"
    end_of_life_date: "2024-12-31"
    
  v1.0.0:
    status: "legacy"
    support_level: "fallback_with_migration"
    deprecation_date: "2024-06-01"
    end_of_life_date: "2025-06-01"
    
  v1.1.0:
    status: "supported"
    support_level: "full_support"
    
  v1.2.0:
    status: "current"
    support_level: "full_support"
```

### **Dynamic Fallback Strategy**

```python
@dataclass
class FallbackSchemaConfig:
    """Configuration for fallback schema handling"""
    
    # Fallback behavior
    fallback_enabled: bool = True
    strict_mode: bool = False  # If True, reject non-conforming traces
    auto_migration_enabled: bool = True
    
    # Legacy support
    supported_legacy_versions: List[str] = field(default_factory=lambda: ["v0.9.0", "v1.0.0", "v1.1.0"])
    legacy_field_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Logging and monitoring
    log_fallback_usage: bool = True
    alert_on_legacy_usage: bool = True
    fallback_usage_threshold: float = 0.1  # Alert if >10% traces use fallback
    
    # Migration settings
    auto_migrate_on_read: bool = True
    migration_batch_size: int = 1000
    migration_rate_limit: int = 100  # per second

class LegacyTraceHandler:
    """Handles legacy trace data with fallback schema"""
    
    def __init__(self, config: FallbackSchemaConfig):
        self.config = config
        self.schema_registry = SchemaRegistry()
        self.migration_engine = TraceMigrationEngine()
        self.fallback_stats = FallbackUsageStats()
    
    async def process_trace(self, raw_trace_data: Dict[str, Any]) -> WorkflowTrace:
        """Process trace data with fallback handling"""
        
        # Detect schema version
        detected_version = self._detect_schema_version(raw_trace_data)
        
        if detected_version == self.schema_registry.current_version:
            # Current schema - direct processing
            return self._process_current_schema(raw_trace_data)
        
        elif detected_version in self.config.supported_legacy_versions:
            # Legacy schema - fallback processing
            return await self._process_legacy_schema(raw_trace_data, detected_version)
        
        else:
            # Unknown or unsupported schema
            return await self._handle_unknown_schema(raw_trace_data, detected_version)
    
    def _detect_schema_version(self, trace_data: Dict[str, Any]) -> str:
        """Detect schema version from trace data"""
        
        # Check explicit version field
        if 'schema_version' in trace_data:
            return trace_data['schema_version']
        
        # Heuristic detection based on field presence
        if 'trust_index' in trace_data:
            return "v1.2.0"  # Current schema has trust_index
        elif 'governance_events' in trace_data:
            return "v1.1.0"  # v1.1.0 introduced governance_events
        elif 'evidence_refs' in trace_data:
            return "v1.0.0"  # v1.0.0 introduced evidence_refs
        else:
            return "v0.9.0"  # Oldest supported version
    
    async def _process_legacy_schema(
        self, 
        trace_data: Dict[str, Any], 
        legacy_version: str
    ) -> WorkflowTrace:
        """Process legacy trace data with fallback schema"""
        
        # Log fallback usage
        if self.config.log_fallback_usage:
            self.fallback_stats.record_fallback_usage(legacy_version)
        
        # Apply field mapping
        mapped_data = self._apply_field_mapping(trace_data, legacy_version)
        
        # Fill missing required fields with defaults
        normalized_data = self._normalize_legacy_data(mapped_data, legacy_version)
        
        # Create trace object
        trace = self._create_trace_from_legacy_data(normalized_data)
        
        # Auto-migration if enabled
        if self.config.auto_migration_enabled:
            await self._schedule_migration(trace.trace_id, legacy_version)
        
        return trace
```

---

## Legacy Schema Definitions

### **v0.9.0 Legacy Schema**

```json
{
  "trace_id": "string",
  "workflow_id": "string", 
  "execution_id": "string",
  "tenant_id": "integer",
  "user_id": "string",
  "status": "string",
  "started_at": "string",
  "completed_at": "string",
  "inputs": {
    "data": "object"
  },
  "outputs": {
    "data": "object"
  },
  "steps": [
    {
      "step_id": "string",
      "step_name": "string",
      "status": "string",
      "inputs": "object",
      "outputs": "object"
    }
  ],
  "error": {
    "code": "string",
    "message": "string"
  }
}
```

### **v1.0.0 Legacy Schema**

```json
{
  "trace_id": "string",
  "workflow_id": "string",
  "execution_id": "string", 
  "workflow_version": "string",
  "tenant_id": "integer",
  "user_id": "string",
  "status": "string",
  "started_at": "string",
  "completed_at": "string",
  "inputs": {
    "input_data": "object",
    "input_hash": "string"
  },
  "outputs": {
    "output_data": "object", 
    "output_hash": "string"
  },
  "steps": [
    {
      "step_id": "string",
      "step_name": "string",
      "step_type": "string",
      "status": "string",
      "inputs": "object",
      "outputs": "object",
      "error": "object"
    }
  ],
  "evidence_refs": ["string"],
  "policy_pack_version": "string",
  "compliance_score": "number"
}
```

### **v1.1.0 Legacy Schema**

```json
{
  "trace_id": "string",
  "workflow_id": "string",
  "execution_id": "string",
  "workflow_version": "string",
  "context": {
    "tenant_id": "integer",
    "user_id": "string",
    "session_id": "string",
    "correlation_id": "string",
    "industry_code": "string"
  },
  "actor": {
    "actor_id": "string",
    "actor_type": "string",
    "actor_name": "string"
  },
  "status": "string",
  "started_at": "string",
  "completed_at": "string",
  "inputs": "object",
  "outputs": "object",
  "steps": ["object"],
  "governance_events": [
    {
      "event_id": "string",
      "policy_id": "string",
      "decision": "string",
      "reason": "string"
    }
  ],
  "evidence_refs": ["string"],
  "trust_score": "number"
}
```

---

## Field Mapping Configuration

### **Dynamic Field Mapping**

```python
class LegacyFieldMapper:
    """Maps legacy fields to current schema"""
    
    def __init__(self):
        self.mapping_rules = {
            "v0.9.0": {
                # Direct field mappings
                "tenant_id": "context.tenant_id",
                "user_id": "context.user_id",
                "inputs.data": "inputs.input_data",
                "outputs.data": "outputs.output_data",
                
                # Default values for missing fields
                "_defaults": {
                    "context.session_id": lambda: str(uuid.uuid4()),
                    "context.correlation_id": lambda: str(uuid.uuid4()),
                    "context.industry_code": "SaaS",
                    "actor.actor_id": lambda data: data.get("user_id", "system"),
                    "actor.actor_type": "user",
                    "actor.actor_name": "legacy_user",
                    "workflow_version": "v1.0.0",
                    "schema_version": "1.2.0",
                    "policy_pack_version": "default_v1.0",
                    "compliance_score": 1.0,
                    "trust_score": 0.8,  # Lower trust for legacy data
                    "trust_index": 0.8
                }
            },
            
            "v1.0.0": {
                # Context restructuring
                "tenant_id": "context.tenant_id", 
                "user_id": "context.user_id",
                
                # New required fields
                "_defaults": {
                    "context.session_id": lambda: str(uuid.uuid4()),
                    "context.correlation_id": lambda: str(uuid.uuid4()),
                    "actor.actor_id": lambda data: data.get("user_id", "system"),
                    "actor.actor_type": "user",
                    "actor.actor_name": "legacy_user",
                    "trust_index": lambda data: data.get("trust_score", 0.8),
                    "trust_factors": {},
                    "trust_calculation_method": "legacy_fallback"
                }
            },
            
            "v1.1.0": {
                # Minor field additions
                "_defaults": {
                    "context.industry_overlay_version": "v1.0",
                    "context.tenant_isolation_level": "standard",
                    "context.pii_redaction_enabled": True,
                    "trust_index": lambda data: data.get("trust_score", 0.8),
                    "trust_factors": {},
                    "override_impact": 0.0
                }
            }
        }
    
    def map_legacy_fields(self, data: Dict[str, Any], version: str) -> Dict[str, Any]:
        """Map legacy fields to current schema format"""
        
        if version not in self.mapping_rules:
            raise UnsupportedSchemaVersionError(f"No mapping rules for version {version}")
        
        rules = self.mapping_rules[version]
        mapped_data = data.copy()
        
        # Apply field mappings
        for legacy_field, current_field in rules.items():
            if legacy_field.startswith("_"):
                continue  # Skip special keys
                
            if legacy_field in data:
                self._set_nested_field(mapped_data, current_field, data[legacy_field])
                # Remove legacy field if it's been moved
                if legacy_field != current_field:
                    self._remove_nested_field(mapped_data, legacy_field)
        
        # Apply default values
        if "_defaults" in rules:
            for field, default_value in rules["_defaults"].items():
                if not self._has_nested_field(mapped_data, field):
                    if callable(default_value):
                        value = default_value(data)
                    else:
                        value = default_value
                    self._set_nested_field(mapped_data, field, value)
        
        return mapped_data
```

### **Data Normalization**

```python
class LegacyDataNormalizer:
    """Normalizes legacy data to meet current schema requirements"""
    
    def normalize_legacy_trace(self, data: Dict[str, Any], version: str) -> Dict[str, Any]:
        """Normalize legacy trace data"""
        
        normalized = data.copy()
        
        # Normalize status values
        normalized = self._normalize_status_values(normalized)
        
        # Normalize timestamps
        normalized = self._normalize_timestamps(normalized)
        
        # Normalize step structure
        normalized = self._normalize_steps(normalized)
        
        # Add governance metadata
        normalized = self._add_governance_metadata(normalized, version)
        
        # Validate required fields
        normalized = self._ensure_required_fields(normalized)
        
        return normalized
    
    def _normalize_status_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize status enum values"""
        
        status_mapping = {
            "success": "completed",
            "failure": "failed", 
            "in_progress": "running",
            "queued": "pending"
        }
        
        if "status" in data and data["status"] in status_mapping:
            data["status"] = status_mapping[data["status"]]
        
        # Normalize step statuses
        if "steps" in data:
            for step in data["steps"]:
                if "status" in step and step["status"] in status_mapping:
                    step["status"] = status_mapping[step["status"]]
        
        return data
    
    def _add_governance_metadata(self, data: Dict[str, Any], version: str) -> Dict[str, Any]:
        """Add governance metadata for legacy traces"""
        
        # Add legacy governance event
        if "governance_events" not in data:
            data["governance_events"] = []
        
        legacy_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "legacy_trace_processed",
            "policy_id": "legacy_compatibility_policy",
            "policy_version": "v1.0",
            "decision": "allowed",
            "reason": f"Legacy trace from schema {version} processed with fallback",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy_applied": True,
            "policy_enforcement_mode": "advisory"
        }
        
        data["governance_events"].append(legacy_event)
        
        return data
```

---

## Migration Engine

### **Automatic Migration**

```python
class TraceMigrationEngine:
    """Handles automatic migration of legacy traces"""
    
    def __init__(self):
        self.migration_queue = asyncio.Queue()
        self.migration_workers = []
        self.migration_stats = MigrationStats()
    
    async def schedule_migration(self, trace_id: str, from_version: str):
        """Schedule trace for migration to current schema"""
        
        migration_task = {
            "trace_id": trace_id,
            "from_version": from_version,
            "to_version": self.current_schema_version,
            "scheduled_at": datetime.now(timezone.utc).isoformat(),
            "priority": self._calculate_migration_priority(from_version)
        }
        
        await self.migration_queue.put(migration_task)
    
    async def migrate_trace(self, trace_id: str, from_version: str) -> Dict[str, Any]:
        """Migrate single trace to current schema"""
        
        try:
            # Load legacy trace
            legacy_trace = await self._load_trace(trace_id)
            
            # Apply migration
            migrated_trace = await self._apply_migration(legacy_trace, from_version)
            
            # Validate migrated trace
            validation_result = await self._validate_migrated_trace(migrated_trace)
            
            if validation_result["valid"]:
                # Store migrated trace
                await self._store_migrated_trace(migrated_trace)
                
                # Update migration stats
                self.migration_stats.record_successful_migration(from_version)
                
                return {
                    "status": "success",
                    "trace_id": trace_id,
                    "from_version": from_version,
                    "to_version": self.current_schema_version
                }
            else:
                # Migration validation failed
                self.migration_stats.record_failed_migration(from_version, validation_result["errors"])
                
                return {
                    "status": "failed",
                    "trace_id": trace_id,
                    "errors": validation_result["errors"]
                }
        
        except Exception as e:
            self.migration_stats.record_migration_error(from_version, str(e))
            raise TraceMigrationError(f"Failed to migrate trace {trace_id}: {str(e)}")
    
    def _calculate_migration_priority(self, from_version: str) -> int:
        """Calculate migration priority based on version age"""
        
        version_priorities = {
            "v0.9.0": 1,  # Highest priority - oldest version
            "v1.0.0": 2,
            "v1.1.0": 3   # Lowest priority - newest legacy version
        }
        
        return version_priorities.get(from_version, 5)
```

### **Batch Migration**

```python
class BatchMigrationManager:
    """Manages batch migration of legacy traces"""
    
    async def migrate_tenant_traces(
        self, 
        tenant_id: int, 
        from_version: str,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Migrate all traces for a tenant from legacy version"""
        
        migration_summary = {
            "tenant_id": tenant_id,
            "from_version": from_version,
            "total_traces": 0,
            "migrated_traces": 0,
            "failed_traces": 0,
            "errors": []
        }
        
        # Get total count
        total_traces = await self._count_legacy_traces(tenant_id, from_version)
        migration_summary["total_traces"] = total_traces
        
        # Process in batches
        for offset in range(0, total_traces, batch_size):
            batch_traces = await self._get_legacy_traces_batch(
                tenant_id, from_version, offset, batch_size
            )
            
            batch_results = await self._migrate_trace_batch(batch_traces)
            
            migration_summary["migrated_traces"] += batch_results["successful"]
            migration_summary["failed_traces"] += batch_results["failed"]
            migration_summary["errors"].extend(batch_results["errors"])
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        return migration_summary
```

---

## Monitoring and Alerting

### **Fallback Usage Monitoring**

```python
class FallbackUsageStats:
    """Tracks fallback schema usage statistics"""
    
    def __init__(self):
        self.usage_stats = {
            "total_traces_processed": 0,
            "fallback_traces_processed": 0,
            "fallback_by_version": {},
            "fallback_rate": 0.0
        }
    
    def record_fallback_usage(self, legacy_version: str):
        """Record usage of fallback schema"""
        
        self.usage_stats["fallback_traces_processed"] += 1
        
        if legacy_version not in self.usage_stats["fallback_by_version"]:
            self.usage_stats["fallback_by_version"][legacy_version] = 0
        
        self.usage_stats["fallback_by_version"][legacy_version] += 1
        
        # Update fallback rate
        if self.usage_stats["total_traces_processed"] > 0:
            self.usage_stats["fallback_rate"] = (
                self.usage_stats["fallback_traces_processed"] / 
                self.usage_stats["total_traces_processed"]
            )
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get fallback usage report"""
        
        return {
            "fallback_rate_percentage": self.usage_stats["fallback_rate"] * 100,
            "total_fallback_traces": self.usage_stats["fallback_traces_processed"],
            "fallback_by_version": self.usage_stats["fallback_by_version"],
            "migration_recommendations": self._get_migration_recommendations()
        }
    
    def _get_migration_recommendations(self) -> List[str]:
        """Get migration recommendations based on usage patterns"""
        
        recommendations = []
        
        for version, count in self.usage_stats["fallback_by_version"].items():
            if count > 1000:
                recommendations.append(
                    f"High usage of legacy version {version} ({count} traces). "
                    f"Consider batch migration."
                )
        
        if self.usage_stats["fallback_rate"] > 0.1:
            recommendations.append(
                f"Fallback rate is {self.usage_stats['fallback_rate']*100:.1f}%. "
                f"Consider accelerating migration efforts."
            )
        
        return recommendations
```

### **Alerting Configuration**

```yaml
fallback_alerts:
  - name: "HighFallbackUsage"
    condition: "fallback_rate > 0.1"
    severity: "warning"
    message: "Fallback schema usage exceeds 10%"
    
  - name: "LegacyVersionDetected"
    condition: "legacy_version_usage{version='v0.9.0'} > 0"
    severity: "info"
    message: "Very old legacy version detected"
    
  - name: "MigrationQueueBacklog"
    condition: "migration_queue_size > 10000"
    severity: "warning"
    message: "Migration queue has large backlog"
    
  - name: "MigrationFailureRate"
    condition: "migration_failure_rate > 0.05"
    severity: "critical"
    message: "Migration failure rate exceeds 5%"
```

---

## Configuration Examples

### **Tenant-Specific Fallback Configuration**

```yaml
# High-priority tenant with strict requirements
tenant_1000_fallback_config:
  fallback_enabled: true
  strict_mode: false
  auto_migration_enabled: true
  supported_legacy_versions: ["v1.1.0", "v1.0.0"]  # No v0.9.0 support
  migration_priority: "high"
  alert_on_fallback: true

# Development tenant with relaxed requirements
tenant_9999_fallback_config:
  fallback_enabled: true
  strict_mode: false
  auto_migration_enabled: false  # Manual migration only
  supported_legacy_versions: ["v0.9.0", "v1.0.0", "v1.1.0"]
  migration_priority: "low"
  alert_on_fallback: false
```

### **Industry-Specific Considerations**

```yaml
# Banking - strict compliance, limited legacy support
banking_fallback_config:
  supported_legacy_versions: ["v1.1.0"]  # Only recent versions
  auto_migration_enabled: true
  migration_priority: "critical"
  compliance_validation_required: true
  
# SaaS - flexible legacy support
saas_fallback_config:
  supported_legacy_versions: ["v0.9.0", "v1.0.0", "v1.1.0"]
  auto_migration_enabled: true
  migration_priority: "standard"
  performance_optimized: true
```

---

## Best Practices

### **Fallback Schema Guidelines**

1. **Minimize Fallback Usage**
   - Actively migrate legacy traces
   - Set deprecation timelines
   - Monitor fallback usage rates

2. **Maintain Data Quality**
   - Validate all fallback data
   - Apply consistent field mappings
   - Preserve audit trails

3. **Performance Considerations**
   - Cache field mapping rules
   - Batch migration operations
   - Monitor processing overhead

4. **Security and Compliance**
   - Apply same governance rules to legacy data
   - Maintain PII redaction for legacy traces
   - Ensure compliance framework coverage

### **Migration Strategy**

1. **Prioritize by Impact**
   - Migrate high-volume tenants first
   - Focus on business-critical traces
   - Consider compliance requirements

2. **Gradual Migration**
   - Start with newest legacy versions
   - Validate migration quality
   - Monitor system performance

3. **Rollback Capability**
   - Maintain original legacy data
   - Support rollback to legacy processing
   - Test rollback procedures

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** platform-engineering@company.com
