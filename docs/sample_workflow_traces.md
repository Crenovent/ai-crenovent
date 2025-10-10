# Sample Workflow Traces

**Tasks 7.1-T31, 7.1-T32, 7.1-T33: Create sample workflow traces**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Engineering Team

---

## Overview

This document provides sample workflow traces for different industry verticals, demonstrating the canonical trace schema in action. These traces serve as examples for developers, compliance officers, and regulators.

---

## 7.1-T31: Sample SaaS Workflow Trace (Pipeline Hygiene)

### Workflow: Pipeline Hygiene Automation
**Industry:** SaaS  
**Use Case:** Automated cleanup of stale opportunities  
**Compliance:** SOX, GDPR

```json
{
  "trace_id": "trace_550e8400-e29b-41d4-a716-446655440001",
  "workflow_id": "wf_pipeline_hygiene_v1_2_0",
  "execution_id": "exec_20241008_143000_001",
  "tenant_id": 1000,
  "industry_overlay": "SaaS",
  "workflow_name": "Pipeline Hygiene - Stale Opportunity Cleanup",
  "workflow_version": "1.2.0",
  "workflow_type": "RBA",
  
  "context": {
    "trigger_type": "scheduled",
    "trigger_source": "cron_daily_0800",
    "execution_mode": "automated",
    "region": "US",
    "environment": "production"
  },
  
  "actor": {
    "actor_type": "system",
    "actor_id": "system_scheduler",
    "user_id": null,
    "role": "automation_service",
    "permissions": ["pipeline_cleanup", "opportunity_update"]
  },
  
  "status": "completed",
  "started_at": "2024-10-08T08:00:00.000Z",
  "completed_at": "2024-10-08T08:02:15.432Z",
  "duration_ms": 135432,
  
  "inputs": {
    "stale_threshold_days": 90,
    "max_opportunities_per_run": 1000,
    "notification_enabled": true,
    "dry_run": false,
    "filters": {
      "stages": ["Prospecting", "Qualification"],
      "min_amount": 0,
      "exclude_accounts": []
    }
  },
  
  "outputs": {
    "opportunities_processed": 247,
    "opportunities_updated": 89,
    "opportunities_closed": 23,
    "notifications_sent": 15,
    "errors": 0,
    "summary": {
      "total_value_cleaned": 1250000.00,
      "accounts_affected": 67,
      "owners_notified": 15
    }
  },
  
  "steps": [
    {
      "step_id": "step_001_query_stale_opps",
      "step_type": "query",
      "step_name": "Query Stale Opportunities",
      "status": "completed",
      "started_at": "2024-10-08T08:00:00.100Z",
      "completed_at": "2024-10-08T08:00:02.543Z",
      "duration_ms": 2443,
      "inputs": {
        "query": "SELECT * FROM opportunities WHERE last_activity_date < NOW() - INTERVAL '90 days' AND stage IN ('Prospecting', 'Qualification')",
        "tenant_id": 1000,
        "limit": 1000
      },
      "outputs": {
        "records_found": 247,
        "query_execution_time_ms": 1234
      }
    },
    {
      "step_id": "step_002_classify_actions",
      "step_type": "decision",
      "step_name": "Classify Required Actions",
      "status": "completed",
      "started_at": "2024-10-08T08:00:02.600Z",
      "completed_at": "2024-10-08T08:00:05.123Z",
      "duration_ms": 2523,
      "inputs": {
        "opportunities": 247,
        "business_rules": {
          "close_threshold_days": 120,
          "update_threshold_days": 90,
          "notify_threshold_days": 60
        }
      },
      "outputs": {
        "to_close": 23,
        "to_update": 89,
        "to_notify": 135
      }
    },
    {
      "step_id": "step_003_update_opportunities",
      "step_type": "query",
      "step_name": "Update Opportunity Records",
      "status": "completed",
      "started_at": "2024-10-08T08:00:05.200Z",
      "completed_at": "2024-10-08T08:01:45.678Z",
      "duration_ms": 100478,
      "inputs": {
        "update_count": 112,
        "batch_size": 50
      },
      "outputs": {
        "records_updated": 112,
        "batches_processed": 3
      }
    },
    {
      "step_id": "step_004_send_notifications",
      "step_type": "notify",
      "step_name": "Notify Opportunity Owners",
      "status": "completed",
      "started_at": "2024-10-08T08:01:45.750Z",
      "completed_at": "2024-10-08T08:02:15.432Z",
      "duration_ms": 29682,
      "inputs": {
        "notification_type": "email",
        "template": "stale_opportunity_cleanup",
        "recipients": 15
      },
      "outputs": {
        "notifications_sent": 15,
        "notifications_failed": 0
      }
    }
  ],
  
  "governance_events": [
    {
      "event_type": "policy_applied",
      "policy_id": "pol_saas_pipeline_hygiene_v1_0",
      "policy_name": "SaaS Pipeline Hygiene Policy",
      "policy_version": "1.0",
      "policy_hash": "sha256:a1b2c3d4e5f6...",
      "applied_at": "2024-10-08T08:00:00.050Z",
      "result": "passed"
    },
    {
      "event_type": "compliance_check",
      "framework": "SOX",
      "check_type": "segregation_of_duties",
      "result": "passed",
      "details": "Automated execution with proper audit trail"
    },
    {
      "event_type": "data_processing",
      "framework": "GDPR",
      "processing_type": "automated_decision",
      "lawful_basis": "legitimate_interest",
      "data_subjects_affected": 0,
      "retention_applied": true
    }
  ],
  
  "evidence_pack_id": "evp_20241008_143000_pipeline_hygiene",
  "evidence_hash": "sha256:f1e2d3c4b5a6...",
  "override_ledger_refs": [],
  
  "trust_score": 1.0,
  "trust_factors": {
    "execution_success": 1.0,
    "policy_compliance": 1.0,
    "data_quality": 0.98,
    "performance": 0.95
  },
  
  "compliance_score": 1.0,
  "compliance_frameworks": ["SOX", "GDPR"],
  
  "resource_usage": {
    "cpu_seconds": 12.5,
    "memory_mb": 256,
    "database_queries": 15,
    "api_calls": 0,
    "storage_mb": 2.1
  },
  
  "performance_metrics": {
    "throughput_records_per_second": 1.82,
    "latency_p95_ms": 2500,
    "error_rate": 0.0,
    "cache_hit_rate": 0.85
  },
  
  "sla_metrics": {
    "execution_time_sla_ms": 300000,
    "execution_time_actual_ms": 135432,
    "sla_met": true,
    "sla_buffer_ms": 164568
  },
  
  "tags": ["pipeline_hygiene", "automated", "saas", "daily"],
  "annotations": {
    "business_impact": "Cleaned $1.25M in stale pipeline value",
    "compliance_note": "Full SOX audit trail maintained",
    "performance_note": "Execution within SLA bounds"
  },
  
  "schema_version": "1.0.0",
  "created_at": "2024-10-08T08:02:15.500Z",
  "checksum": "sha256:trace_checksum_abc123..."
}
```

---

## 7.1-T32: Sample Banking Workflow Trace (Loan Sanction Check)

### Workflow: Loan Sanction Compliance Check
**Industry:** Banking  
**Use Case:** Automated compliance validation for loan sanctions  
**Compliance:** RBI, SOX, BASEL

```json
{
  "trace_id": "trace_550e8400-e29b-41d4-a716-446655440002",
  "workflow_id": "wf_loan_sanction_check_v2_1_0",
  "execution_id": "exec_20241008_143100_002",
  "tenant_id": 1001,
  "industry_overlay": "BANK",
  "workflow_name": "Loan Sanction Compliance Validation",
  "workflow_version": "2.1.0",
  "workflow_type": "RBA",
  
  "context": {
    "trigger_type": "event",
    "trigger_source": "loan_application_submitted",
    "execution_mode": "real_time",
    "region": "IN",
    "environment": "production",
    "regulatory_context": {
      "rbi_guidelines": "2024_lending_norms",
      "basel_framework": "basel_iii",
      "local_regulations": ["sarfaesi_act", "banking_regulation_act"]
    }
  },
  
  "actor": {
    "actor_type": "user",
    "actor_id": "user_1234",
    "user_id": 1234,
    "role": "loan_officer",
    "permissions": ["loan_review", "compliance_check", "sanction_initiate"],
    "branch_code": "BR001",
    "employee_id": "EMP001234"
  },
  
  "status": "completed",
  "started_at": "2024-10-08T08:31:00.000Z",
  "completed_at": "2024-10-08T08:31:45.123Z",
  "duration_ms": 45123,
  
  "inputs": {
    "loan_application_id": "LA_2024_001234",
    "applicant_id": "CUST_567890",
    "loan_amount": 5000000.00,
    "loan_type": "business_loan",
    "tenure_months": 60,
    "collateral_value": 7500000.00,
    "applicant_details": {
      "credit_score": 750,
      "annual_income": 12000000.00,
      "existing_loans": 2,
      "kyc_status": "completed",
      "aml_status": "cleared"
    }
  },
  
  "outputs": {
    "compliance_status": "approved",
    "risk_rating": "medium",
    "recommended_interest_rate": 12.5,
    "conditions": [
      "Quarterly financial statements required",
      "Collateral insurance mandatory",
      "Personal guarantee from director"
    ],
    "next_steps": ["credit_committee_review", "legal_documentation"],
    "compliance_checks_passed": 15,
    "compliance_checks_failed": 0
  },
  
  "steps": [
    {
      "step_id": "step_001_kyc_verification",
      "step_type": "query",
      "step_name": "KYC Verification Check",
      "status": "completed",
      "started_at": "2024-10-08T08:31:00.100Z",
      "completed_at": "2024-10-08T08:31:05.234Z",
      "duration_ms": 5134,
      "inputs": {
        "applicant_id": "CUST_567890",
        "verification_type": "enhanced_kyc"
      },
      "outputs": {
        "kyc_status": "verified",
        "risk_category": "low",
        "verification_date": "2024-09-15"
      }
    },
    {
      "step_id": "step_002_credit_assessment",
      "step_type": "ml_decision",
      "step_name": "Credit Risk Assessment",
      "status": "completed",
      "started_at": "2024-10-08T08:31:05.300Z",
      "completed_at": "2024-10-08T08:31:25.567Z",
      "duration_ms": 20267,
      "inputs": {
        "credit_score": 750,
        "income_verification": true,
        "collateral_ratio": 1.5,
        "existing_obligations": 2400000.00
      },
      "outputs": {
        "risk_score": 0.25,
        "probability_of_default": 0.03,
        "recommended_ltv": 0.67,
        "model_version": "credit_risk_v3_2"
      }
    },
    {
      "step_id": "step_003_regulatory_compliance",
      "step_type": "decision",
      "step_name": "RBI Compliance Validation",
      "status": "completed",
      "started_at": "2024-10-08T08:31:25.600Z",
      "completed_at": "2024-10-08T08:31:40.890Z",
      "duration_ms": 15290,
      "inputs": {
        "loan_amount": 5000000.00,
        "borrower_category": "msme",
        "sector": "manufacturing",
        "rbi_guidelines": "2024_lending_norms"
      },
      "outputs": {
        "compliance_status": "compliant",
        "applicable_guidelines": ["msme_lending", "sector_exposure"],
        "exposure_limits_check": "passed",
        "documentation_requirements": ["financial_statements", "project_report"]
      }
    },
    {
      "step_id": "step_004_generate_sanction_letter",
      "step_type": "notify",
      "step_name": "Generate Sanction Documentation",
      "status": "completed",
      "started_at": "2024-10-08T08:31:40.950Z",
      "completed_at": "2024-10-08T08:31:45.123Z",
      "duration_ms": 4173,
      "inputs": {
        "template": "business_loan_sanction",
        "loan_details": "approved_terms",
        "regulatory_disclosures": true
      },
      "outputs": {
        "sanction_letter_id": "SL_2024_001234",
        "document_generated": true,
        "regulatory_disclosures_included": true
      }
    }
  ],
  
  "governance_events": [
    {
      "event_type": "policy_applied",
      "policy_id": "pol_rbi_lending_compliance_v2_0",
      "policy_name": "RBI Lending Compliance Policy",
      "policy_version": "2.0",
      "policy_hash": "sha256:rbi_policy_hash_123...",
      "applied_at": "2024-10-08T08:31:00.050Z",
      "result": "passed"
    },
    {
      "event_type": "regulatory_check",
      "framework": "RBI",
      "check_type": "exposure_limits",
      "result": "passed",
      "details": "Sector exposure within prescribed limits"
    },
    {
      "event_type": "compliance_check",
      "framework": "BASEL_III",
      "check_type": "capital_adequacy",
      "result": "passed",
      "details": "Risk weighted assets calculation compliant"
    },
    {
      "event_type": "audit_trail",
      "framework": "SOX",
      "event": "loan_sanction_decision",
      "approver": "user_1234",
      "timestamp": "2024-10-08T08:31:45.123Z"
    }
  ],
  
  "evidence_pack_id": "evp_20241008_143100_loan_sanction",
  "evidence_hash": "sha256:banking_evidence_hash_456...",
  "override_ledger_refs": [],
  
  "trust_score": 0.95,
  "trust_factors": {
    "execution_success": 1.0,
    "policy_compliance": 1.0,
    "regulatory_compliance": 1.0,
    "data_quality": 0.92,
    "model_confidence": 0.88
  },
  
  "compliance_score": 1.0,
  "compliance_frameworks": ["RBI", "SOX", "BASEL_III"],
  
  "resource_usage": {
    "cpu_seconds": 8.2,
    "memory_mb": 512,
    "database_queries": 25,
    "api_calls": 5,
    "storage_mb": 1.8
  },
  
  "performance_metrics": {
    "throughput_applications_per_hour": 80,
    "latency_p95_ms": 50000,
    "error_rate": 0.0,
    "cache_hit_rate": 0.75
  },
  
  "sla_metrics": {
    "execution_time_sla_ms": 60000,
    "execution_time_actual_ms": 45123,
    "sla_met": true,
    "sla_buffer_ms": 14877
  },
  
  "tags": ["loan_sanction", "rbi_compliance", "banking", "real_time"],
  "annotations": {
    "business_impact": "₹50L loan application processed with full compliance",
    "regulatory_note": "All RBI guidelines followed, audit trail complete",
    "risk_note": "Medium risk rating with appropriate conditions"
  },
  
  "schema_version": "1.0.0",
  "created_at": "2024-10-08T08:31:45.200Z",
  "checksum": "sha256:banking_trace_checksum_def456..."
}
```

---

## 7.1-T33: Sample Insurance Workflow Trace (Claim Solvency Check)

### Workflow: Insurance Claim Solvency Validation
**Industry:** Insurance  
**Use Case:** Automated solvency check for large claims  
**Compliance:** IRDAI, SOX, Solvency II

```json
{
  "trace_id": "trace_550e8400-e29b-41d4-a716-446655440003",
  "workflow_id": "wf_claim_solvency_check_v1_5_0",
  "execution_id": "exec_20241008_143200_003",
  "tenant_id": 1002,
  "industry_overlay": "INSUR",
  "workflow_name": "Large Claim Solvency Impact Assessment",
  "workflow_version": "1.5.0",
  "workflow_type": "RBA",
  
  "context": {
    "trigger_type": "event",
    "trigger_source": "claim_amount_threshold_exceeded",
    "execution_mode": "real_time",
    "region": "IN",
    "environment": "production",
    "regulatory_context": {
      "irdai_guidelines": "2024_solvency_norms",
      "solvency_framework": "solvency_ii_equivalent",
      "local_regulations": ["insurance_act_1938", "irdai_regulations_2013"]
    }
  },
  
  "actor": {
    "actor_type": "user",
    "actor_id": "user_5678",
    "user_id": 5678,
    "role": "claims_manager",
    "permissions": ["claim_review", "solvency_check", "large_claim_approval"],
    "branch_code": "INS_MUM_001",
    "employee_id": "EMP005678"
  },
  
  "status": "completed",
  "started_at": "2024-10-08T08:32:00.000Z",
  "completed_at": "2024-10-08T08:32:35.789Z",
  "duration_ms": 35789,
  
  "inputs": {
    "claim_id": "CLM_2024_007890",
    "policy_id": "POL_2023_123456",
    "claim_amount": 25000000.00,
    "claim_type": "property_damage",
    "incident_date": "2024-10-05",
    "policy_details": {
      "sum_insured": 50000000.00,
      "premium_paid": 500000.00,
      "policy_start_date": "2023-01-01",
      "policy_end_date": "2024-12-31"
    },
    "current_reserves": {
      "technical_reserves": 15000000000.00,
      "solvency_capital": 2500000000.00,
      "available_capital": 3000000000.00
    }
  },
  
  "outputs": {
    "solvency_impact": "acceptable",
    "solvency_ratio_before": 1.25,
    "solvency_ratio_after": 1.24,
    "capital_impact": 25000000.00,
    "recommendation": "approve_with_conditions",
    "conditions": [
      "Independent loss adjuster assessment required",
      "Reinsurance recovery to be pursued",
      "Monthly solvency monitoring for next quarter"
    ],
    "reinsurance_recovery_expected": 15000000.00,
    "net_impact": 10000000.00
  },
  
  "steps": [
    {
      "step_id": "step_001_policy_validation",
      "step_type": "query",
      "step_name": "Policy Coverage Validation",
      "status": "completed",
      "started_at": "2024-10-08T08:32:00.100Z",
      "completed_at": "2024-10-08T08:32:05.432Z",
      "duration_ms": 5332,
      "inputs": {
        "policy_id": "POL_2023_123456",
        "claim_type": "property_damage",
        "claim_amount": 25000000.00
      },
      "outputs": {
        "coverage_valid": true,
        "coverage_amount": 50000000.00,
        "deductible": 500000.00,
        "policy_status": "active"
      }
    },
    {
      "step_id": "step_002_solvency_calculation",
      "step_type": "ml_decision",
      "step_name": "Solvency Impact Calculation",
      "status": "completed",
      "started_at": "2024-10-08T08:32:05.500Z",
      "completed_at": "2024-10-08T08:32:20.123Z",
      "duration_ms": 14623,
      "inputs": {
        "current_solvency_ratio": 1.25,
        "claim_amount": 25000000.00,
        "reinsurance_treaties": ["quota_share_50", "surplus_treaty"],
        "capital_base": 3000000000.00
      },
      "outputs": {
        "projected_solvency_ratio": 1.24,
        "capital_impact": 25000000.00,
        "reinsurance_recovery": 15000000.00,
        "net_capital_impact": 10000000.00,
        "model_version": "solvency_impact_v2_1"
      }
    },
    {
      "step_id": "step_003_regulatory_compliance",
      "step_type": "decision",
      "step_name": "IRDAI Solvency Compliance Check",
      "status": "completed",
      "started_at": "2024-10-08T08:32:20.200Z",
      "completed_at": "2024-10-08T08:32:30.567Z",
      "duration_ms": 10367,
      "inputs": {
        "projected_solvency_ratio": 1.24,
        "minimum_solvency_ratio": 1.50,
        "regulatory_buffer": 0.10,
        "irdai_guidelines": "2024_solvency_norms"
      },
      "outputs": {
        "compliance_status": "non_compliant",
        "required_action": "board_approval_required",
        "regulatory_notification": "required_within_24h",
        "capital_augmentation_needed": false
      }
    },
    {
      "step_id": "step_004_risk_mitigation",
      "step_type": "decision",
      "step_name": "Risk Mitigation Assessment",
      "status": "completed",
      "started_at": "2024-10-08T08:32:30.600Z",
      "completed_at": "2024-10-08T08:32:35.789Z",
      "duration_ms": 5189,
      "inputs": {
        "solvency_shortfall": 0.26,
        "available_mitigations": ["reinsurance_recovery", "capital_injection", "claim_settlement_terms"]
      },
      "outputs": {
        "recommended_mitigations": [
          "Pursue reinsurance recovery aggressively",
          "Negotiate phased settlement",
          "Monitor solvency daily"
        ],
        "timeline": "30_days",
        "escalation_required": true
      }
    }
  ],
  
  "governance_events": [
    {
      "event_type": "policy_applied",
      "policy_id": "pol_irdai_solvency_compliance_v1_0",
      "policy_name": "IRDAI Solvency Compliance Policy",
      "policy_version": "1.0",
      "policy_hash": "sha256:irdai_policy_hash_789...",
      "applied_at": "2024-10-08T08:32:00.050Z",
      "result": "warning"
    },
    {
      "event_type": "regulatory_check",
      "framework": "IRDAI",
      "check_type": "solvency_ratio",
      "result": "below_threshold",
      "details": "Solvency ratio 1.24 below minimum 1.50, board approval required"
    },
    {
      "event_type": "compliance_check",
      "framework": "SOLVENCY_II",
      "check_type": "capital_adequacy",
      "result": "warning",
      "details": "Capital impact significant, monitoring required"
    },
    {
      "event_type": "escalation",
      "framework": "SOX",
      "event": "large_claim_board_approval",
      "escalated_to": "board_of_directors",
      "timestamp": "2024-10-08T08:32:35.789Z"
    }
  ],
  
  "evidence_pack_id": "evp_20241008_143200_claim_solvency",
  "evidence_hash": "sha256:insurance_evidence_hash_789...",
  "override_ledger_refs": [],
  
  "trust_score": 0.85,
  "trust_factors": {
    "execution_success": 1.0,
    "policy_compliance": 0.7,
    "regulatory_compliance": 0.8,
    "data_quality": 0.95,
    "model_confidence": 0.92
  },
  
  "compliance_score": 0.8,
  "compliance_frameworks": ["IRDAI", "SOX", "SOLVENCY_II"],
  
  "resource_usage": {
    "cpu_seconds": 6.8,
    "memory_mb": 384,
    "database_queries": 18,
    "api_calls": 3,
    "storage_mb": 2.5
  },
  
  "performance_metrics": {
    "throughput_claims_per_hour": 100,
    "latency_p95_ms": 40000,
    "error_rate": 0.0,
    "cache_hit_rate": 0.80
  },
  
  "sla_metrics": {
    "execution_time_sla_ms": 45000,
    "execution_time_actual_ms": 35789,
    "sla_met": true,
    "sla_buffer_ms": 9211
  },
  
  "tags": ["claim_solvency", "irdai_compliance", "insurance", "large_claim"],
  "annotations": {
    "business_impact": "₹2.5Cr claim requires board approval due to solvency impact",
    "regulatory_note": "IRDAI notification required within 24 hours",
    "risk_note": "Solvency ratio below minimum, mitigation plan activated"
  },
  
  "schema_version": "1.0.0",
  "created_at": "2024-10-08T08:32:35.900Z",
  "checksum": "sha256:insurance_trace_checksum_ghi789..."
}
```

---

## Usage Guidelines

### For Developers
- Use these traces as templates for implementing new workflows
- Follow the canonical schema structure exactly
- Ensure all required fields are populated
- Include appropriate governance events for your industry

### For Compliance Officers
- Review governance events and compliance scores
- Verify that regulatory frameworks are properly referenced
- Ensure evidence pack linkage is maintained
- Check that audit trails are complete

### For Regulators
- These traces demonstrate full audit capability
- All policy applications and compliance checks are logged
- Evidence packs provide immutable audit trails
- Cross-references enable full traceability

---

## Schema Validation

All sample traces conform to the canonical trace schema v1.0 and include:

✅ **Required Fields**: All mandatory fields populated  
✅ **Governance Events**: Policy applications and compliance checks  
✅ **Evidence Linkage**: Evidence pack references included  
✅ **Industry Overlays**: Industry-specific compliance frameworks  
✅ **Trust Scoring**: Trust factors and compliance scores  
✅ **Performance Metrics**: SLA and resource usage tracking  
✅ **Audit Trail**: Complete execution timeline with checksums  

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025
