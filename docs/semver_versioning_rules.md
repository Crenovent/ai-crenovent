# Semantic Versioning Rules for Workflow Registry

**Task 7.3-T02: Choose versioning scheme (SemVer) & rules**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Architecture Team

---

## Overview

This document defines the semantic versioning (SemVer) rules and policies for the Workflow Registry. All workflows, policy packs, and related artifacts must follow these versioning conventions to ensure predictable upgrades, compatibility management, and safe deployments across multi-tenant environments.

---

## Semantic Versioning Specification

### **Version Format**
```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

**Examples:**
- `1.0.0` - Initial stable release
- `1.2.3` - Standard release
- `2.0.0-alpha.1` - Pre-release version
- `1.5.2+build.123` - Release with build metadata
- `3.1.0-beta.2+exp.sha.5114f85` - Complex pre-release

### **Version Components**

#### **MAJOR Version (X.y.z)**
- **Increment when**: Making incompatible API changes
- **Breaking changes include**:
  - Removing or renaming workflow parameters
  - Changing parameter types or validation rules
  - Removing workflow steps or capabilities
  - Changing workflow behavior in incompatible ways
  - Modifying governance requirements that affect existing executions

#### **MINOR Version (x.Y.z)**
- **Increment when**: Adding functionality in a backward-compatible manner
- **Compatible changes include**:
  - Adding new optional parameters
  - Adding new workflow steps or capabilities
  - Enhancing existing functionality without breaking changes
  - Adding new governance policies (non-breaking)
  - Improving performance or reliability

#### **PATCH Version (x.y.Z)**
- **Increment when**: Making backward-compatible bug fixes
- **Bug fixes include**:
  - Fixing incorrect behavior
  - Correcting validation logic
  - Resolving security vulnerabilities
  - Improving error messages
  - Documentation corrections

#### **PRE-RELEASE Identifiers**
- **Format**: `-alpha.N`, `-beta.N`, `-rc.N`
- **Purpose**: Testing and validation before stable release
- **Precedence**: `1.0.0-alpha.1 < 1.0.0-alpha.2 < 1.0.0-beta.1 < 1.0.0-rc.1 < 1.0.0`

#### **BUILD Metadata**
- **Format**: `+build.N`, `+sha.HASH`, `+timestamp.YYYYMMDD`
- **Purpose**: Build identification and traceability
- **Precedence**: Build metadata does not affect version precedence

---

## Versioning Rules by Artifact Type

### **1. Workflow Versions**

#### **Breaking Changes (MAJOR)**
```yaml
breaking_changes:
  parameter_changes:
    - removing_required_parameter: "Remove required input parameter"
    - changing_parameter_type: "Change parameter from string to integer"
    - renaming_parameter: "Rename 'account_id' to 'customer_id'"
    - adding_required_parameter: "Add new required parameter without default"
  
  behavior_changes:
    - changing_output_format: "Change output from array to object"
    - removing_workflow_step: "Remove email notification step"
    - changing_execution_order: "Reorder steps in incompatible way"
    - modifying_side_effects: "Change database write behavior"
  
  governance_changes:
    - stricter_policies: "Add mandatory approval requirement"
    - removing_capabilities: "Remove admin override capability"
    - changing_permissions: "Change required role from user to admin"
```

#### **Compatible Changes (MINOR)**
```yaml
compatible_changes:
  enhancements:
    - adding_optional_parameter: "Add optional 'priority' parameter"
    - adding_workflow_step: "Add optional logging step"
    - improving_validation: "Add better input validation"
    - adding_output_fields: "Add metadata to output object"
  
  governance_additions:
    - optional_policies: "Add optional compliance check"
    - additional_capabilities: "Add new admin capabilities"
    - enhanced_logging: "Add detailed audit logging"
```

#### **Bug Fixes (PATCH)**
```yaml
bug_fixes:
  corrections:
    - fixing_validation_logic: "Fix email validation regex"
    - correcting_calculations: "Fix percentage calculation error"
    - resolving_edge_cases: "Handle null input values properly"
    - improving_error_messages: "Clarify validation error messages"
  
  security_fixes:
    - patching_vulnerabilities: "Fix SQL injection vulnerability"
    - updating_dependencies: "Update vulnerable dependency"
    - improving_sanitization: "Enhance input sanitization"
```

### **2. Policy Pack Versions**

#### **Breaking Changes (MAJOR)**
```yaml
policy_breaking_changes:
  enforcement_changes:
    - stricter_enforcement: "Change from advisory to strict enforcement"
    - removing_exemptions: "Remove emergency override capability"
    - changing_scope: "Apply policy to all tenants (was opt-in)"
  
  rule_changes:
    - modifying_conditions: "Change approval threshold from 1 to 2 people"
    - removing_rules: "Remove deprecated compliance rule"
    - changing_validation: "Stricter data validation requirements"
```

#### **Compatible Changes (MINOR)**
```yaml
policy_compatible_changes:
  additions:
    - new_optional_rules: "Add optional GDPR compliance check"
    - enhanced_reporting: "Add detailed compliance reporting"
    - additional_exemptions: "Add emergency bypass capability"
  
  improvements:
    - better_error_messages: "Improve policy violation messages"
    - performance_optimization: "Optimize policy evaluation"
```

### **3. Connector Versions**

#### **Breaking Changes (MAJOR)**
```yaml
connector_breaking_changes:
  interface_changes:
    - changing_api_contract: "Modify required authentication method"
    - removing_endpoints: "Remove deprecated API endpoint"
    - changing_data_format: "Change response format from XML to JSON"
  
  configuration_changes:
    - required_config_changes: "Add mandatory API key configuration"
    - removing_config_options: "Remove deprecated configuration option"
```

---

## Version Lifecycle Management

### **Version States**
```yaml
version_states:
  draft:
    description: "Work in progress, not ready for use"
    allowed_operations: ["edit", "delete", "promote_to_review"]
    restrictions: ["cannot_be_published", "cannot_be_used"]
  
  review:
    description: "Under review for approval"
    allowed_operations: ["approve", "reject", "edit_with_reset"]
    restrictions: ["cannot_be_published", "cannot_be_used"]
  
  approved:
    description: "Approved for publication"
    allowed_operations: ["publish", "edit_with_reset"]
    restrictions: ["cannot_be_used_until_published"]
  
  published:
    description: "Available for use"
    allowed_operations: ["deprecate", "create_patch"]
    restrictions: ["immutable", "cannot_edit"]
  
  deprecated:
    description: "Discouraged from use, will be retired"
    allowed_operations: ["retire", "create_patch"]
    restrictions: ["warning_on_use"]
  
  retired:
    description: "No longer supported"
    allowed_operations: ["archive"]
    restrictions: ["cannot_be_used", "read_only"]
  
  archived:
    description: "Preserved for compliance/audit"
    allowed_operations: ["read"]
    restrictions: ["read_only", "compliance_retention_only"]
```

### **Version Promotion Rules**
```yaml
promotion_rules:
  draft_to_review:
    requirements:
      - "All required fields completed"
      - "Basic validation passes"
      - "No critical linting errors"
    
  review_to_approved:
    requirements:
      - "Two-person approval rule satisfied"
      - "Security scan passes"
      - "Compliance check passes"
      - "Breaking change analysis completed"
    
  approved_to_published:
    requirements:
      - "Final integration tests pass"
      - "Compatibility matrix validated"
      - "Documentation complete"
      - "SBOM generated"
```

---

## Version Aliases and Channels

### **Standard Aliases**
```yaml
version_aliases:
  latest:
    description: "Most recent published version"
    update_policy: "automatic"
    stability: "may_include_breaking_changes"
    
  stable:
    description: "Latest stable version (no pre-release)"
    update_policy: "automatic_stable_only"
    stability: "backward_compatible_within_major"
    
  lts:
    description: "Long-term support version"
    update_policy: "manual_promotion"
    stability: "extended_support_lifecycle"
    support_duration: "24_months"
    
  canary:
    description: "Pre-release testing version"
    update_policy: "automatic_prerelease"
    stability: "experimental"
    usage_restriction: "testing_environments_only"
```

### **Channel Management**
```yaml
release_channels:
  alpha:
    purpose: "Early development testing"
    audience: "internal_developers"
    update_frequency: "continuous"
    stability_guarantee: "none"
    
  beta:
    purpose: "Feature testing and validation"
    audience: "selected_customers"
    update_frequency: "weekly"
    stability_guarantee: "api_stable"
    
  rc:
    purpose: "Release candidate validation"
    audience: "all_customers_opt_in"
    update_frequency: "as_needed"
    stability_guarantee: "production_ready"
    
  stable:
    purpose: "Production use"
    audience: "all_customers"
    update_frequency: "monthly"
    stability_guarantee: "full_support"
```

---

## Compatibility Management

### **Compatibility Matrix**
```yaml
compatibility_rules:
  runtime_compatibility:
    major_version_changes:
      - "May require runtime upgrade"
      - "Breaking changes allowed"
      - "Migration guide required"
    
    minor_version_changes:
      - "Backward compatible with same major"
      - "May require feature flags"
      - "Optional runtime upgrade"
    
    patch_version_changes:
      - "Fully backward compatible"
      - "No runtime changes required"
      - "Safe automatic upgrade"
  
  dependency_compatibility:
    connector_versions:
      - "Pin to compatible major versions"
      - "Allow minor/patch updates"
      - "Test compatibility matrix"
    
    policy_versions:
      - "Maintain policy compatibility"
      - "Version policy bindings"
      - "Support graceful upgrades"
```

### **Deprecation Policy**
```yaml
deprecation_policy:
  deprecation_notice:
    advance_notice: "90_days_minimum"
    communication_channels: ["email", "slack", "dashboard", "api_headers"]
    documentation_updates: "required"
    
  deprecation_timeline:
    major_versions:
      support_duration: "18_months"
      deprecation_notice: "6_months"
      retirement_notice: "3_months"
      
    minor_versions:
      support_duration: "12_months"
      deprecation_notice: "3_months"
      retirement_notice: "1_month"
      
    patch_versions:
      support_duration: "6_months"
      deprecation_notice: "1_month"
      retirement_notice: "2_weeks"
  
  migration_support:
    migration_guides: "required_for_breaking_changes"
    automated_migration: "provided_when_possible"
    support_assistance: "available_during_transition"
```

---

## Version Numbering Strategies

### **Initial Versions**
```yaml
initial_versioning:
  new_workflow:
    first_version: "0.1.0"
    development_phase: "0.x.y"
    first_stable: "1.0.0"
    
  forked_workflow:
    version_strategy: "inherit_major_increment_minor"
    example: "parent_v2.3.1 → fork_v2.4.0"
    
  migrated_workflow:
    version_strategy: "reset_to_1.0.0"
    migration_note: "required_in_changelog"
```

### **Branching Strategy**
```yaml
version_branches:
  main_branch:
    name: "main"
    version_type: "development"
    auto_increment: "patch"
    
  release_branches:
    naming: "release/v{major}.{minor}"
    purpose: "stabilization_and_patches"
    merge_policy: "cherry_pick_from_main"
    
  hotfix_branches:
    naming: "hotfix/v{major}.{minor}.{patch}"
    purpose: "critical_bug_fixes"
    merge_policy: "merge_to_release_and_main"
```

---

## Automated Version Management

### **Version Calculation Rules**
```python
def calculate_next_version(current_version: str, change_type: str) -> str:
    """
    Calculate next version based on change type
    """
    major, minor, patch = parse_version(current_version)
    
    if change_type == "breaking":
        return f"{major + 1}.0.0"
    elif change_type == "feature":
        return f"{major}.{minor + 1}.0"
    elif change_type == "bugfix":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Unknown change type: {change_type}")

def validate_version_increment(current: str, proposed: str) -> bool:
    """
    Validate that version increment follows SemVer rules
    """
    current_parts = parse_version(current)
    proposed_parts = parse_version(proposed)
    
    # Ensure version only increments by one component
    # and resets lower components appropriately
    return is_valid_semver_increment(current_parts, proposed_parts)
```

### **Change Detection**
```yaml
automated_change_detection:
  breaking_changes:
    detection_methods:
      - "api_schema_comparison"
      - "parameter_analysis"
      - "behavior_testing"
      - "static_analysis"
    
    confidence_levels:
      - "high_confidence_auto_increment"
      - "medium_confidence_flag_for_review"
      - "low_confidence_require_manual_review"
  
  compatible_changes:
    detection_methods:
      - "additive_change_detection"
      - "enhancement_analysis"
      - "performance_improvement_detection"
    
    auto_increment_policy: "enabled_with_review"
```

---

## Version Validation Rules

### **Format Validation**
```regex
# Valid version format
^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$

# Pre-release identifiers
alpha\.\d+|beta\.\d+|rc\.\d+

# Build metadata
build\.\d+|sha\.[a-f0-9]+|timestamp\.\d{8}
```

### **Business Rules Validation**
```yaml
validation_rules:
  version_uniqueness:
    - "Version must be unique within workflow"
    - "Cannot reuse retired version numbers"
    - "Pre-release versions must be sequential"
  
  increment_validation:
    - "Must increment exactly one version component"
    - "Lower components must reset to 0 on higher increment"
    - "Cannot skip version numbers"
  
  lifecycle_validation:
    - "Cannot publish version lower than current published"
    - "Cannot deprecate version that is not published"
    - "Cannot retire version that is not deprecated"
```

---

## Integration with Registry

### **Version Storage**
```sql
-- Example version storage schema
CREATE TABLE workflow_versions (
    version_id UUID PRIMARY KEY,
    workflow_id UUID NOT NULL,
    version_number VARCHAR(50) NOT NULL,
    major_version INTEGER NOT NULL,
    minor_version INTEGER NOT NULL,
    patch_version INTEGER NOT NULL,
    pre_release VARCHAR(50),
    build_metadata VARCHAR(100),
    status version_status NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    
    CONSTRAINT uq_workflow_version UNIQUE (workflow_id, version_number),
    CONSTRAINT chk_version_format CHECK (
        version_number ~ '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)?(?:\+[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)?$'
    )
);
```

### **Version Resolution API**
```yaml
version_resolution:
  exact_version:
    endpoint: "GET /workflows/{id}/versions/{version}"
    example: "GET /workflows/abc123/versions/1.2.3"
    
  alias_resolution:
    endpoint: "GET /workflows/{id}/versions/{alias}"
    examples:
      - "GET /workflows/abc123/versions/latest"
      - "GET /workflows/abc123/versions/stable"
      - "GET /workflows/abc123/versions/lts"
  
  range_resolution:
    endpoint: "GET /workflows/{id}/versions?range={semver_range}"
    examples:
      - "GET /workflows/abc123/versions?range=^1.2.0"
      - "GET /workflows/abc123/versions?range=~1.2.3"
      - "GET /workflows/abc123/versions?range=>=1.0.0 <2.0.0"
```

---

## Monitoring and Analytics

### **Version Metrics**
```yaml
version_analytics:
  adoption_metrics:
    - "version_download_count"
    - "active_usage_by_version"
    - "upgrade_adoption_rate"
    - "time_to_upgrade"
  
  quality_metrics:
    - "version_success_rate"
    - "bug_report_count_by_version"
    - "rollback_frequency"
    - "support_ticket_volume"
  
  lifecycle_metrics:
    - "time_in_each_state"
    - "approval_time"
    - "deprecation_compliance"
    - "retirement_timeline_adherence"
```

### **Version Health Dashboard**
```yaml
dashboard_metrics:
  version_distribution:
    - "percentage_on_latest"
    - "percentage_on_deprecated"
    - "percentage_on_retired"
    
  upgrade_trends:
    - "upgrade_velocity"
    - "breaking_change_impact"
    - "migration_success_rate"
    
  quality_indicators:
    - "version_stability_score"
    - "compatibility_success_rate"
    - "rollback_incidents"
```

---

## Best Practices

### **For Workflow Authors**
1. **Plan Breaking Changes**: Group breaking changes into major releases
2. **Communicate Early**: Announce deprecations well in advance
3. **Provide Migration Paths**: Include clear upgrade instructions
4. **Test Compatibility**: Validate against supported runtime versions
5. **Document Changes**: Maintain comprehensive changelogs

### **For Platform Operators**
1. **Monitor Version Health**: Track adoption and quality metrics
2. **Enforce Policies**: Implement automated validation rules
3. **Support Migrations**: Provide tooling and assistance for upgrades
4. **Manage Lifecycle**: Actively deprecate and retire old versions
5. **Maintain Security**: Keep dependencies updated and scan for vulnerabilities

### **For Consumers**
1. **Pin Versions**: Use specific versions in production
2. **Test Upgrades**: Validate new versions in staging environments
3. **Stay Current**: Regularly upgrade to supported versions
4. **Monitor Deprecations**: Subscribe to deprecation notifications
5. **Plan Migrations**: Allocate time for major version upgrades

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** platform-architecture@company.com
