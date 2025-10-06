-- ============================================================================
-- CHAPTER 15: ROUTING ORCHESTRATOR DATABASE SCHEMA
-- ============================================================================
-- Implements all database tables for Chapter 15 sections:
-- 15.1: Intent Parsing (intent_registry_*)
-- 15.2: Policy Gate (policy_gate_*)
-- 15.3: Capability Lookup (capability_registry_*)
-- 15.4: Plan Synthesis (plan_synth_*)
-- 15.5: Dispatcher (dispatcher_exec_*)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- SECTION 15.1: INTENT PARSING TABLES
-- ============================================================================

-- Task 15.1.1: Provision intent_registry_intent table (Store raw intents)
CREATE TABLE IF NOT EXISTS intent_registry_intent (
    intent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Raw Intent Data
    raw_input TEXT NOT NULL,
    input_source VARCHAR(50) NOT NULL, -- ui, api, agent, webhook, scheduled
    session_id VARCHAR(255),
    user_id INTEGER,
    
    -- Context Information
    context_data JSONB DEFAULT '{}',
    user_agent TEXT,
    ip_address INET,
    request_headers JSONB DEFAULT '{}',
    
    -- Industry & SLA Context
    industry_code VARCHAR(20),
    sla_tier VARCHAR(10),
    tenant_tier VARCHAR(10),
    
    -- Timestamps
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    
    -- Governance
    compliance_tags JSONB DEFAULT '[]',
    data_classification VARCHAR(20) DEFAULT 'internal',
    
    CONSTRAINT valid_input_source CHECK (input_source IN ('ui', 'api', 'agent', 'webhook', 'scheduled')),
    CONSTRAINT valid_sla_tier CHECK (sla_tier IN ('T0', 'T1', 'T2')),
    CONSTRAINT valid_tenant_tier CHECK (tenant_tier IN ('T0', 'T1', 'T2')),
    CONSTRAINT valid_data_classification CHECK (data_classification IN ('public', 'internal', 'confidential', 'restricted')),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Task 15.1.2: Provision intent_registry_parse table (Store parsed outputs)
CREATE TABLE IF NOT EXISTS intent_registry_parse (
    parse_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    intent_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Parsing Results
    parsed_intent_type VARCHAR(50) NOT NULL, -- workflow_execution, capability_discovery, help_request, invalid
    automation_route VARCHAR(10), -- RBA, RBIA, AALA
    workflow_category VARCHAR(100),
    confidence_score DECIMAL(5,4) NOT NULL,
    
    -- Extracted Parameters
    extracted_parameters JSONB DEFAULT '{}',
    parameter_confidence JSONB DEFAULT '{}',
    
    -- Classification Details
    classifier_model_id VARCHAR(100),
    classifier_version VARCHAR(20),
    classification_method VARCHAR(50), -- regex, ml_model, hybrid, fallback
    
    -- Industry-Specific Classification
    saas_intent_category VARCHAR(50), -- arr_analysis, churn_prediction, comp_plan, qbr
    banking_intent_category VARCHAR(50), -- credit_scoring, aml_detection, npa_tracking, fraud
    insurance_intent_category VARCHAR(50), -- claims_processing, underwriting, phi_compliance
    
    -- Parsing Performance
    parsing_duration_ms INTEGER,
    parsing_status VARCHAR(20) NOT NULL DEFAULT 'completed',
    
    -- Timestamps
    parsed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Governance
    parsing_evidence JSONB DEFAULT '{}',
    
    CONSTRAINT valid_intent_type CHECK (parsed_intent_type IN ('workflow_execution', 'capability_discovery', 'help_request', 'invalid')),
    CONSTRAINT valid_automation_route CHECK (automation_route IN ('RBA', 'RBIA', 'AALA')),
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT valid_parsing_status CHECK (parsing_status IN ('pending', 'completed', 'failed', 'timeout')),
    FOREIGN KEY (intent_id) REFERENCES intent_registry_intent(intent_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Task 15.1.3: Provision intent_registry_reason table (Store reason codes)
CREATE TABLE IF NOT EXISTS intent_registry_reason (
    reason_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parse_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Reasoning Information
    reason_type VARCHAR(30) NOT NULL, -- policy_override, model_prediction, regex_match, fallback_rule
    reason_code VARCHAR(100) NOT NULL,
    reason_description TEXT,
    
    -- Decision Factors
    decision_factors JSONB DEFAULT '{}',
    policy_pack_id VARCHAR(255),
    rule_id VARCHAR(100),
    
    -- Confidence & Weight
    reason_confidence DECIMAL(5,4),
    decision_weight DECIMAL(5,4),
    
    -- Traceability
    model_explanation JSONB DEFAULT '{}',
    feature_importance JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_reason_type CHECK (reason_type IN ('policy_override', 'model_prediction', 'regex_match', 'fallback_rule')),
    CONSTRAINT valid_reason_confidence CHECK (reason_confidence IS NULL OR (reason_confidence >= 0.0 AND reason_confidence <= 1.0)),
    CONSTRAINT valid_decision_weight CHECK (decision_weight IS NULL OR (decision_weight >= 0.0 AND decision_weight <= 1.0)),
    FOREIGN KEY (parse_id) REFERENCES intent_registry_parse(parse_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- ============================================================================
-- SECTION 15.2: POLICY GATE TABLES
-- ============================================================================

-- Task 15.2.1: Provision policy_gate_rule table (Store policy rules applied at gate)
CREATE TABLE IF NOT EXISTS policy_gate_rule (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Rule Definition
    rule_name VARCHAR(255) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- access_control, data_residency, compliance_check, sla_enforcement
    rule_category VARCHAR(50) NOT NULL, -- saas, banking, insurance, general
    
    -- Rule Content
    rule_definition JSONB NOT NULL,
    opa_policy_text TEXT,
    rule_conditions JSONB DEFAULT '{}',
    
    -- Industry & Compliance
    industry_overlay VARCHAR(20),
    compliance_frameworks JSONB DEFAULT '[]', -- ["SOX", "GDPR", "RBI", "HIPAA"]
    jurisdiction VARCHAR(10), -- US, EU, IN, etc.
    
    -- Rule Metadata
    rule_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    rule_status VARCHAR(20) NOT NULL DEFAULT 'active',
    enforcement_level VARCHAR(20) NOT NULL DEFAULT 'strict', -- strict, advisory, disabled
    
    -- Performance
    avg_evaluation_time_ms INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 1.0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    effective_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    effective_until TIMESTAMPTZ,
    
    -- Governance
    created_by_user_id INTEGER,
    approved_by_user_id INTEGER,
    
    CONSTRAINT valid_rule_type CHECK (rule_type IN ('access_control', 'data_residency', 'compliance_check', 'sla_enforcement')),
    CONSTRAINT valid_rule_category CHECK (rule_category IN ('saas', 'banking', 'insurance', 'general')),
    CONSTRAINT valid_rule_status CHECK (rule_status IN ('draft', 'active', 'deprecated', 'retired')),
    CONSTRAINT valid_enforcement_level CHECK (enforcement_level IN ('strict', 'advisory', 'disabled')),
    CONSTRAINT valid_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    UNIQUE (tenant_id, rule_name, rule_version)
);

-- Task 15.2.2: Provision policy_gate_log table (Store enforcement events)
CREATE TABLE IF NOT EXISTS policy_gate_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parse_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Enforcement Event
    enforcement_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    rule_id UUID NOT NULL,
    
    -- Enforcement Result
    enforcement_result VARCHAR(20) NOT NULL, -- allowed, denied, warning
    enforcement_reason TEXT,
    policy_violations JSONB DEFAULT '[]',
    
    -- Decision Details
    decision_factors JSONB DEFAULT '{}',
    evaluation_duration_ms INTEGER,
    opa_decision JSONB DEFAULT '{}',
    
    -- Context
    request_context JSONB DEFAULT '{}',
    user_context JSONB DEFAULT '{}',
    tenant_context JSONB DEFAULT '{}',
    
    -- Timestamps
    enforced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Governance
    evidence_pack_id UUID,
    audit_trail_id UUID,
    
    CONSTRAINT valid_enforcement_result CHECK (enforcement_result IN ('allowed', 'denied', 'warning')),
    FOREIGN KEY (parse_id) REFERENCES intent_registry_parse(parse_id),
    FOREIGN KEY (rule_id) REFERENCES policy_gate_rule(rule_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Task 15.2.3: Provision policy_gate_override table (Capture override references)
CREATE TABLE IF NOT EXISTS policy_gate_override (
    override_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    log_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Override Details
    override_reason TEXT NOT NULL,
    override_type VARCHAR(30) NOT NULL, -- emergency, business_exception, technical_issue
    override_duration_hours INTEGER,
    
    -- Approval Workflow
    requested_by_user_id INTEGER NOT NULL,
    approved_by_user_id INTEGER,
    approval_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    
    -- Business Impact
    business_justification TEXT,
    risk_assessment TEXT,
    impact_level VARCHAR(20) NOT NULL, -- low, medium, high, critical
    
    -- Timestamps
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    
    -- Governance
    override_ledger_id UUID,
    evidence_pack_id UUID,
    
    CONSTRAINT valid_override_type CHECK (override_type IN ('emergency', 'business_exception', 'technical_issue')),
    CONSTRAINT valid_approval_status CHECK (approval_status IN ('pending', 'approved', 'denied', 'expired', 'revoked')),
    CONSTRAINT valid_impact_level CHECK (impact_level IN ('low', 'medium', 'high', 'critical')),
    FOREIGN KEY (log_id) REFERENCES policy_gate_log(log_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- ============================================================================
-- SECTION 15.3: CAPABILITY LOOKUP TABLES
-- ============================================================================

-- Task 15.3.1: Provision capability_registry_meta table (Store capability metadata)
CREATE TABLE IF NOT EXISTS capability_registry_meta (
    capability_meta_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Capability Identity
    capability_name VARCHAR(255) NOT NULL,
    capability_type VARCHAR(50) NOT NULL, -- RBA_TEMPLATE, RBIA_MODEL, AALA_AGENT
    capability_category VARCHAR(100) NOT NULL, -- pipeline, forecasting, planning, compliance
    
    -- Industry & Context
    industry_tags JSONB DEFAULT '[]', -- ["SaaS", "Banking", "Insurance"]
    persona_tags JSONB DEFAULT '[]', -- ["CRO", "Sales Manager", "RevOps"]
    use_case_tags JSONB DEFAULT '[]', -- ["arr_analysis", "churn_prediction"]
    
    -- Capability Definition
    capability_description TEXT,
    input_schema JSONB DEFAULT '{}',
    output_schema JSONB DEFAULT '{}',
    configuration_schema JSONB DEFAULT '{}',
    
    -- Performance Characteristics
    avg_execution_time_ms INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 1.0,
    trust_score DECIMAL(5,4) DEFAULT 1.0,
    
    -- Cost & SLA
    estimated_cost_per_execution DECIMAL(10,6) DEFAULT 0.0,
    sla_tier VARCHAR(10) DEFAULT 'T2',
    max_concurrent_executions INTEGER DEFAULT 10,
    
    -- Lifecycle
    readiness_state VARCHAR(20) NOT NULL DEFAULT 'draft', -- draft, beta, certified, deprecated
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    
    -- Governance
    created_by_user_id INTEGER,
    owner_team VARCHAR(100),
    compliance_tags JSONB DEFAULT '[]',
    
    CONSTRAINT valid_capability_type CHECK (capability_type IN ('RBA_TEMPLATE', 'RBIA_MODEL', 'AALA_AGENT')),
    CONSTRAINT valid_readiness_state CHECK (readiness_state IN ('draft', 'beta', 'certified', 'deprecated')),
    CONSTRAINT valid_sla_tier CHECK (sla_tier IN ('T0', 'T1', 'T2')),
    CONSTRAINT valid_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    CONSTRAINT valid_trust_score CHECK (trust_score >= 0.0 AND trust_score <= 1.0),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    UNIQUE (tenant_id, capability_name, version)
);

-- Task 15.3.2: Provision capability_registry_version table (Store version lifecycle)
CREATE TABLE IF NOT EXISTS capability_registry_version (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    capability_meta_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Version Information
    version_number VARCHAR(50) NOT NULL,
    version_status VARCHAR(20) NOT NULL DEFAULT 'draft', -- draft, testing, promoted, deprecated, retired
    
    -- Version Content
    capability_definition JSONB NOT NULL,
    version_hash VARCHAR(64) NOT NULL,
    
    -- Change Information
    change_description TEXT,
    breaking_changes BOOLEAN DEFAULT false,
    migration_notes TEXT,
    
    -- Lifecycle Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tested_at TIMESTAMPTZ,
    promoted_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    
    -- Governance
    created_by_user_id INTEGER,
    approved_by_user_id INTEGER,
    
    CONSTRAINT valid_version_status CHECK (version_status IN ('draft', 'testing', 'promoted', 'deprecated', 'retired')),
    FOREIGN KEY (capability_meta_id) REFERENCES capability_registry_meta(capability_meta_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    UNIQUE (capability_meta_id, version_number)
);

-- Task 15.3.3: Provision capability_registry_binding table (Map capabilities to tenants/industries)
CREATE TABLE IF NOT EXISTS capability_registry_binding (
    binding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    capability_meta_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Binding Configuration
    binding_type VARCHAR(30) NOT NULL, -- tenant_specific, industry_shared, global_shared
    binding_status VARCHAR(20) NOT NULL DEFAULT 'active',
    
    -- Access Control
    allowed_user_roles JSONB DEFAULT '[]',
    allowed_personas JSONB DEFAULT '[]',
    access_restrictions JSONB DEFAULT '{}',
    
    -- Industry & Context Filters
    industry_restrictions JSONB DEFAULT '[]',
    sla_tier_restrictions JSONB DEFAULT '[]',
    compliance_requirements JSONB DEFAULT '[]',
    
    -- Usage Limits
    max_executions_per_hour INTEGER,
    max_executions_per_day INTEGER,
    max_concurrent_executions INTEGER DEFAULT 5,
    
    -- Cost Controls
    max_cost_per_execution DECIMAL(10,6),
    monthly_budget_limit DECIMAL(12,2),
    
    -- Timestamps
    bound_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_accessed_at TIMESTAMPTZ,
    
    -- Governance
    bound_by_user_id INTEGER,
    approved_by_user_id INTEGER,
    
    CONSTRAINT valid_binding_type CHECK (binding_type IN ('tenant_specific', 'industry_shared', 'global_shared')),
    CONSTRAINT valid_binding_status CHECK (binding_status IN ('active', 'suspended', 'expired')),
    FOREIGN KEY (capability_meta_id) REFERENCES capability_registry_meta(capability_meta_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    UNIQUE (capability_meta_id, tenant_id)
);

-- ============================================================================
-- SECTION 15.4: PLAN SYNTHESIS TABLES
-- ============================================================================

-- Task 15.4.1: Provision plan_synth_meta table (Store plan metadata)
CREATE TABLE IF NOT EXISTS plan_synth_meta (
    plan_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parse_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Plan Identity
    plan_name VARCHAR(255),
    plan_type VARCHAR(30) NOT NULL, -- single_capability, hybrid_workflow, multi_step_plan
    plan_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    
    -- Plan Configuration
    execution_strategy VARCHAR(30) NOT NULL, -- sequential, parallel, conditional
    plan_definition JSONB NOT NULL,
    plan_hash VARCHAR(64) NOT NULL,
    
    -- Cost & Performance Estimates
    estimated_total_cost DECIMAL(12,6) DEFAULT 0.0,
    estimated_total_duration_ms INTEGER DEFAULT 0,
    estimated_resource_usage JSONB DEFAULT '{}',
    
    -- SLA & Quality
    target_sla_tier VARCHAR(10),
    quality_score DECIMAL(5,4),
    risk_score DECIMAL(5,4),
    
    -- Plan Status
    synthesis_status VARCHAR(20) NOT NULL DEFAULT 'synthesized',
    validation_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    approval_status VARCHAR(20) NOT NULL DEFAULT 'auto_approved',
    
    -- Timestamps
    synthesized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    validated_at TIMESTAMPTZ,
    approved_at TIMESTAMPTZ,
    
    -- Governance
    synthesized_by_user_id INTEGER,
    approved_by_user_id INTEGER,
    evidence_pack_id UUID,
    
    CONSTRAINT valid_plan_type CHECK (plan_type IN ('single_capability', 'hybrid_workflow', 'multi_step_plan')),
    CONSTRAINT valid_execution_strategy CHECK (execution_strategy IN ('sequential', 'parallel', 'conditional')),
    CONSTRAINT valid_synthesis_status CHECK (synthesis_status IN ('synthesizing', 'synthesized', 'failed')),
    CONSTRAINT valid_validation_status CHECK (validation_status IN ('pending', 'validated', 'failed')),
    CONSTRAINT valid_approval_status CHECK (approval_status IN ('auto_approved', 'pending', 'approved', 'denied')),
    CONSTRAINT valid_target_sla_tier CHECK (target_sla_tier IN ('T0', 'T1', 'T2')),
    CONSTRAINT valid_quality_score CHECK (quality_score IS NULL OR (quality_score >= 0.0 AND quality_score <= 1.0)),
    CONSTRAINT valid_risk_score CHECK (risk_score IS NULL OR (risk_score >= 0.0 AND risk_score <= 1.0)),
    FOREIGN KEY (parse_id) REFERENCES intent_registry_parse(parse_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Task 15.4.2: Provision plan_synth_step table (Store plan steps)
CREATE TABLE IF NOT EXISTS plan_synth_step (
    step_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Step Identity
    step_name VARCHAR(255) NOT NULL,
    step_order INTEGER NOT NULL,
    step_type VARCHAR(30) NOT NULL, -- capability_execution, decision_point, parallel_branch, loop
    
    -- Step Configuration
    capability_meta_id UUID,
    step_configuration JSONB DEFAULT '{}',
    input_mapping JSONB DEFAULT '{}',
    output_mapping JSONB DEFAULT '{}',
    
    -- Execution Details
    automation_type VARCHAR(10), -- RBA, RBIA, AALA
    execution_mode VARCHAR(20) DEFAULT 'normal', -- normal, retry, fallback
    
    -- Dependencies
    depends_on_steps JSONB DEFAULT '[]', -- Array of step_ids
    parallel_group VARCHAR(50),
    
    -- Cost & Performance
    estimated_cost DECIMAL(10,6) DEFAULT 0.0,
    estimated_duration_ms INTEGER DEFAULT 0,
    timeout_ms INTEGER DEFAULT 300000, -- 5 minutes default
    
    -- Error Handling
    retry_policy JSONB DEFAULT '{}',
    fallback_step_id UUID,
    error_handling_strategy VARCHAR(30) DEFAULT 'fail_fast',
    
    -- Governance
    policy_requirements JSONB DEFAULT '[]',
    compliance_checks JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_step_type CHECK (step_type IN ('capability_execution', 'decision_point', 'parallel_branch', 'loop')),
    CONSTRAINT valid_automation_type CHECK (automation_type IN ('RBA', 'RBIA', 'AALA')),
    CONSTRAINT valid_execution_mode CHECK (execution_mode IN ('normal', 'retry', 'fallback')),
    CONSTRAINT valid_error_handling CHECK (error_handling_strategy IN ('fail_fast', 'continue', 'fallback')),
    FOREIGN KEY (plan_id) REFERENCES plan_synth_meta(plan_id),
    FOREIGN KEY (capability_meta_id) REFERENCES capability_registry_meta(capability_meta_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    UNIQUE (plan_id, step_order)
);

-- Task 15.4.3: Provision plan_synth_manifest table (Store signed execution manifests)
CREATE TABLE IF NOT EXISTS plan_synth_manifest (
    manifest_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Manifest Content
    manifest_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    manifest_content JSONB NOT NULL,
    manifest_hash VARCHAR(64) NOT NULL,
    
    -- Digital Signature
    signature_algorithm VARCHAR(50) DEFAULT 'SHA256withRSA',
    digital_signature TEXT,
    signing_certificate TEXT,
    
    -- Immutable Anchoring
    blockchain_anchor VARCHAR(128),
    immudb_proof TEXT,
    
    -- Manifest Metadata
    manifest_type VARCHAR(30) NOT NULL DEFAULT 'execution_plan',
    compliance_attestation JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signed_at TIMESTAMPTZ,
    anchored_at TIMESTAMPTZ,
    
    -- Governance
    created_by_user_id INTEGER,
    signed_by_user_id INTEGER,
    
    CONSTRAINT valid_manifest_type CHECK (manifest_type IN ('execution_plan', 'compliance_attestation', 'audit_trail')),
    FOREIGN KEY (plan_id) REFERENCES plan_synth_meta(plan_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    UNIQUE (plan_id, manifest_version)
);

-- ============================================================================
-- SECTION 15.5: DISPATCHER TABLES
-- ============================================================================

-- Task 15.5.1: Provision dispatcher_exec_meta table (Store execution metadata)
CREATE TABLE IF NOT EXISTS dispatcher_exec_meta (
    execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Execution Identity
    execution_name VARCHAR(255),
    execution_type VARCHAR(30) NOT NULL, -- workflow_execution, capability_test, plan_validation
    execution_mode VARCHAR(20) NOT NULL DEFAULT 'production', -- production, test, debug
    
    -- Execution Status
    execution_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    execution_progress DECIMAL(5,2) DEFAULT 0.0, -- 0.0 to 100.0
    current_step_id UUID,
    
    -- Performance Metrics
    actual_duration_ms INTEGER,
    actual_cost DECIMAL(12,6) DEFAULT 0.0,
    resource_usage JSONB DEFAULT '{}',
    
    -- SLA Tracking
    sla_tier VARCHAR(10),
    sla_target_duration_ms INTEGER,
    sla_compliance BOOLEAN,
    sla_breach_reason TEXT,
    
    -- Error Information
    error_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    error_details JSONB DEFAULT '{}',
    
    -- Retry & Recovery
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    recovery_attempts INTEGER DEFAULT 0,
    
    -- Timestamps
    scheduled_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    
    -- Governance
    executed_by_user_id INTEGER,
    evidence_pack_id UUID,
    audit_trail_id UUID,
    
    CONSTRAINT valid_execution_type CHECK (execution_type IN ('workflow_execution', 'capability_test', 'plan_validation')),
    CONSTRAINT valid_execution_mode CHECK (execution_mode IN ('production', 'test', 'debug')),
    CONSTRAINT valid_execution_status CHECK (execution_status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout')),
    CONSTRAINT valid_execution_progress CHECK (execution_progress >= 0.0 AND execution_progress <= 100.0),
    CONSTRAINT valid_sla_tier CHECK (sla_tier IN ('T0', 'T1', 'T2')),
    FOREIGN KEY (plan_id) REFERENCES plan_synth_meta(plan_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Task 15.5.2: Provision dispatcher_exec_step table (Store step-level execution logs)
CREATE TABLE IF NOT EXISTS dispatcher_exec_step (
    step_execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL,
    plan_step_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Step Execution Details
    step_name VARCHAR(255) NOT NULL,
    step_order INTEGER NOT NULL,
    execution_attempt INTEGER DEFAULT 1,
    
    -- Step Status
    step_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    step_progress DECIMAL(5,2) DEFAULT 0.0,
    
    -- Input/Output Data
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    intermediate_data JSONB DEFAULT '{}',
    
    -- Performance
    step_duration_ms INTEGER,
    step_cost DECIMAL(10,6) DEFAULT 0.0,
    resource_consumption JSONB DEFAULT '{}',
    
    -- Error Handling
    error_message TEXT,
    error_code VARCHAR(50),
    error_details JSONB DEFAULT '{}',
    retry_reason TEXT,
    
    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    
    -- Governance
    step_evidence JSONB DEFAULT '{}',
    policy_evaluations JSONB DEFAULT '[]',
    
    CONSTRAINT valid_step_status CHECK (step_status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'timeout')),
    CONSTRAINT valid_step_progress CHECK (step_progress >= 0.0 AND step_progress <= 100.0),
    FOREIGN KEY (execution_id) REFERENCES dispatcher_exec_meta(execution_id),
    FOREIGN KEY (plan_step_id) REFERENCES plan_synth_step(step_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    UNIQUE (execution_id, plan_step_id, execution_attempt)
);

-- Task 15.5.3: Provision dispatcher_exec_result table (Store execution results & evidence)
CREATE TABLE IF NOT EXISTS dispatcher_exec_result (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Result Summary
    result_type VARCHAR(30) NOT NULL, -- success, partial_success, failure, timeout
    result_summary TEXT,
    result_data JSONB DEFAULT '{}',
    
    -- Business Impact
    business_outcome VARCHAR(50),
    business_value DECIMAL(12,2),
    user_satisfaction_score DECIMAL(3,2),
    
    -- Quality Metrics
    accuracy_score DECIMAL(5,4),
    completeness_score DECIMAL(5,4),
    timeliness_score DECIMAL(5,4),
    overall_quality_score DECIMAL(5,4),
    
    -- Trust & Reliability
    trust_score_before DECIMAL(5,4),
    trust_score_after DECIMAL(5,4),
    reliability_impact DECIMAL(5,4),
    
    -- Evidence & Compliance
    evidence_pack_id UUID,
    compliance_validation JSONB DEFAULT '{}',
    audit_trail JSONB DEFAULT '{}',
    
    -- Feedback & Learning
    user_feedback JSONB DEFAULT '{}',
    system_feedback JSONB DEFAULT '{}',
    improvement_suggestions JSONB DEFAULT '[]',
    
    -- Timestamps
    result_generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    evidence_collected_at TIMESTAMPTZ,
    
    -- Governance
    validated_by_user_id INTEGER,
    approved_by_user_id INTEGER,
    
    CONSTRAINT valid_result_type CHECK (result_type IN ('success', 'partial_success', 'failure', 'timeout')),
    CONSTRAINT valid_user_satisfaction CHECK (user_satisfaction_score IS NULL OR (user_satisfaction_score >= 0.0 AND user_satisfaction_score <= 5.0)),
    CONSTRAINT valid_accuracy_score CHECK (accuracy_score IS NULL OR (accuracy_score >= 0.0 AND accuracy_score <= 1.0)),
    CONSTRAINT valid_completeness_score CHECK (completeness_score IS NULL OR (completeness_score >= 0.0 AND completeness_score <= 1.0)),
    CONSTRAINT valid_timeliness_score CHECK (timeliness_score IS NULL OR (timeliness_score >= 0.0 AND timeliness_score <= 1.0)),
    CONSTRAINT valid_overall_quality_score CHECK (overall_quality_score IS NULL OR (overall_quality_score >= 0.0 AND overall_quality_score <= 1.0)),
    CONSTRAINT valid_trust_score_before CHECK (trust_score_before IS NULL OR (trust_score_before >= 0.0 AND trust_score_before <= 1.0)),
    CONSTRAINT valid_trust_score_after CHECK (trust_score_after IS NULL OR (trust_score_after >= 0.0 AND trust_score_after <= 1.0)),
    CONSTRAINT valid_reliability_impact CHECK (reliability_impact IS NULL OR (reliability_impact >= -1.0 AND reliability_impact <= 1.0)),
    FOREIGN KEY (execution_id) REFERENCES dispatcher_exec_meta(execution_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    UNIQUE (execution_id)
);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE intent_registry_intent ENABLE ROW LEVEL SECURITY;
ALTER TABLE intent_registry_parse ENABLE ROW LEVEL SECURITY;
ALTER TABLE intent_registry_reason ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy_gate_rule ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy_gate_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy_gate_override ENABLE ROW LEVEL SECURITY;
ALTER TABLE capability_registry_meta ENABLE ROW LEVEL SECURITY;
ALTER TABLE capability_registry_version ENABLE ROW LEVEL SECURITY;
ALTER TABLE capability_registry_binding ENABLE ROW LEVEL SECURITY;
ALTER TABLE plan_synth_meta ENABLE ROW LEVEL SECURITY;
ALTER TABLE plan_synth_step ENABLE ROW LEVEL SECURITY;
ALTER TABLE plan_synth_manifest ENABLE ROW LEVEL SECURITY;
ALTER TABLE dispatcher_exec_meta ENABLE ROW LEVEL SECURITY;
ALTER TABLE dispatcher_exec_step ENABLE ROW LEVEL SECURITY;
ALTER TABLE dispatcher_exec_result ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
CREATE POLICY intent_registry_intent_rls_policy ON intent_registry_intent
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY intent_registry_parse_rls_policy ON intent_registry_parse
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY intent_registry_reason_rls_policy ON intent_registry_reason
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY policy_gate_rule_rls_policy ON policy_gate_rule
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY policy_gate_log_rls_policy ON policy_gate_log
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY policy_gate_override_rls_policy ON policy_gate_override
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY capability_registry_meta_rls_policy ON capability_registry_meta
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY capability_registry_version_rls_policy ON capability_registry_version
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY capability_registry_binding_rls_policy ON capability_registry_binding
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY plan_synth_meta_rls_policy ON plan_synth_meta
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY plan_synth_step_rls_policy ON plan_synth_step
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY plan_synth_manifest_rls_policy ON plan_synth_manifest
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY dispatcher_exec_meta_rls_policy ON dispatcher_exec_meta
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY dispatcher_exec_step_rls_policy ON dispatcher_exec_step
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY dispatcher_exec_result_rls_policy ON dispatcher_exec_result
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- PERFORMANCE INDEXES
-- ============================================================================

-- Intent Parsing Indexes
CREATE INDEX IF NOT EXISTS idx_intent_registry_intent_tenant_received ON intent_registry_intent(tenant_id, received_at);
CREATE INDEX IF NOT EXISTS idx_intent_registry_intent_source_industry ON intent_registry_intent(input_source, industry_code);
CREATE INDEX IF NOT EXISTS idx_intent_registry_parse_tenant_confidence ON intent_registry_parse(tenant_id, confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_intent_registry_parse_automation_route ON intent_registry_parse(automation_route, parsed_at);

-- Policy Gate Indexes
CREATE INDEX IF NOT EXISTS idx_policy_gate_rule_tenant_status ON policy_gate_rule(tenant_id, rule_status);
CREATE INDEX IF NOT EXISTS idx_policy_gate_rule_industry_compliance ON policy_gate_rule(industry_overlay, compliance_frameworks);
CREATE INDEX IF NOT EXISTS idx_policy_gate_log_tenant_result ON policy_gate_log(tenant_id, enforcement_result);
CREATE INDEX IF NOT EXISTS idx_policy_gate_log_enforced_at ON policy_gate_log(enforced_at);

-- Capability Lookup Indexes
CREATE INDEX IF NOT EXISTS idx_capability_registry_meta_tenant_type ON capability_registry_meta(tenant_id, capability_type);
CREATE INDEX IF NOT EXISTS idx_capability_registry_meta_industry_tags ON capability_registry_meta USING GIN(industry_tags);
CREATE INDEX IF NOT EXISTS idx_capability_registry_meta_readiness_trust ON capability_registry_meta(readiness_state, trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_capability_registry_binding_tenant_status ON capability_registry_binding(tenant_id, binding_status);

-- Plan Synthesis Indexes
CREATE INDEX IF NOT EXISTS idx_plan_synth_meta_tenant_status ON plan_synth_meta(tenant_id, synthesis_status);
CREATE INDEX IF NOT EXISTS idx_plan_synth_meta_synthesized_at ON plan_synth_meta(synthesized_at);
CREATE INDEX IF NOT EXISTS idx_plan_synth_step_plan_order ON plan_synth_step(plan_id, step_order);
CREATE INDEX IF NOT EXISTS idx_plan_synth_step_automation_type ON plan_synth_step(automation_type, step_type);

-- Dispatcher Indexes
CREATE INDEX IF NOT EXISTS idx_dispatcher_exec_meta_tenant_status ON dispatcher_exec_meta(tenant_id, execution_status);
CREATE INDEX IF NOT EXISTS idx_dispatcher_exec_meta_started_at ON dispatcher_exec_meta(started_at);
CREATE INDEX IF NOT EXISTS idx_dispatcher_exec_meta_sla_compliance ON dispatcher_exec_meta(sla_tier, sla_compliance);
CREATE INDEX IF NOT EXISTS idx_dispatcher_exec_step_execution_order ON dispatcher_exec_step(execution_id, step_order);
CREATE INDEX IF NOT EXISTS idx_dispatcher_exec_result_tenant_type ON dispatcher_exec_result(tenant_id, result_type);

-- ============================================================================
-- SAMPLE DATA FOR SAAS INDUSTRY
-- ============================================================================

-- Insert sample SaaS intent taxonomies
INSERT INTO intent_registry_intent (tenant_id, raw_input, input_source, industry_code, sla_tier, tenant_tier, context_data) VALUES
(1300, 'analyze ARR growth for Q4 2024', 'ui', 'SAAS', 'T1', 'T1', '{"persona": "CRO", "module": "forecasting"}'),
(1300, 'predict churn risk for enterprise accounts', 'api', 'SAAS', 'T1', 'T1', '{"persona": "RevOps", "module": "pipeline"}'),
(1300, 'generate compensation plan for sales team', 'ui', 'SAAS', 'T2', 'T1', '{"persona": "Sales Manager", "module": "compensation"}'),
(1300, 'create QBR presentation for Microsoft account', 'ui', 'SAAS', 'T1', 'T1', '{"persona": "AE", "module": "planning"}')
ON CONFLICT DO NOTHING;

-- Insert sample SaaS capabilities
INSERT INTO capability_registry_meta (
    tenant_id, capability_name, capability_type, capability_category,
    industry_tags, persona_tags, use_case_tags, capability_description,
    avg_execution_time_ms, success_rate, trust_score, estimated_cost_per_execution,
    sla_tier, readiness_state, version, owner_team
) VALUES
(1300, 'ARR Growth Analyzer', 'RBA_TEMPLATE', 'forecasting', 
 '["SaaS"]', '["CRO", "RevOps"]', '["arr_analysis", "growth_tracking"]',
 'Automated ARR growth analysis with trend prediction and variance detection',
 15000, 0.95, 0.92, 12.50, 'T1', 'certified', '2.1.0', 'RevOps Team'),

(1300, 'Churn Prediction Engine', 'RBIA_MODEL', 'pipeline',
 '["SaaS"]', '["RevOps", "Customer Success"]', '["churn_prediction", "retention"]',
 'ML-powered churn prediction with customer health scoring and intervention recommendations',
 45000, 0.88, 0.85, 35.75, 'T1', 'certified', '1.8.2', 'AI Team'),

(1300, 'Compensation Plan Builder', 'RBA_TEMPLATE', 'compensation',
 '["SaaS"]', '["Sales Manager", "Finance"]', '["comp_plan", "quota_management"]',
 'Automated compensation plan generation with quota allocation and performance tracking',
 25000, 0.92, 0.89, 18.25, 'T2', 'certified', '1.5.1', 'Finance Team'),

(1300, 'QBR Assistant', 'AALA_AGENT', 'planning',
 '["SaaS"]', '["AE", "Account Manager"]', '["qbr", "account_planning"]',
 'AI-powered QBR preparation with account insights, growth opportunities, and risk assessment',
 120000, 0.78, 0.82, 85.50, 'T1', 'beta', '0.9.3', 'AI Team')
ON CONFLICT DO NOTHING;

-- Insert sample policy gate rules for SaaS
INSERT INTO policy_gate_rule (
    tenant_id, rule_name, rule_type, rule_category, rule_definition,
    industry_overlay, compliance_frameworks, jurisdiction, enforcement_level,
    created_by_user_id
) VALUES
(1300, 'SaaS SOX Revenue Recognition', 'compliance_check', 'saas',
 '{"rule": "revenue_data_access", "conditions": {"requires_sox_approval": true, "data_classification": "financial"}}',
 'SAAS', '["SOX"]', 'US', 'strict', 1323),

(1300, 'SaaS GDPR Customer Data', 'data_residency', 'saas',
 '{"rule": "customer_data_processing", "conditions": {"eu_residents": true, "consent_required": true}}',
 'SAAS', '["GDPR"]', 'EU', 'strict', 1323),

(1300, 'SaaS Subscription Lifecycle', 'access_control', 'saas',
 '{"rule": "subscription_operations", "conditions": {"role_required": ["RevOps", "Finance"], "approval_threshold": 10000}}',
 'SAAS', '["SAAS_BUSINESS_RULES"]', 'US', 'advisory', 1323)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE intent_registry_intent IS 'Raw user intents with context and metadata for routing orchestrator';
COMMENT ON TABLE intent_registry_parse IS 'Parsed intent results with automation route classification and confidence scores';
COMMENT ON TABLE intent_registry_reason IS 'Reasoning and decision factors for intent parsing with traceability';
COMMENT ON TABLE policy_gate_rule IS 'Policy rules for compliance enforcement at the routing gate';
COMMENT ON TABLE policy_gate_log IS 'Immutable log of policy enforcement events and decisions';
COMMENT ON TABLE policy_gate_override IS 'Override requests and approvals for policy violations';
COMMENT ON TABLE capability_registry_meta IS 'Capability metadata with performance and trust metrics';
COMMENT ON TABLE capability_registry_version IS 'Version lifecycle management for capabilities';
COMMENT ON TABLE capability_registry_binding IS 'Tenant-specific capability bindings and access controls';
COMMENT ON TABLE plan_synth_meta IS 'Execution plan metadata with cost estimates and SLA targets';
COMMENT ON TABLE plan_synth_step IS 'Individual steps in execution plans with dependencies and error handling';
COMMENT ON TABLE plan_synth_manifest IS 'Signed and anchored execution manifests for audit trails';
COMMENT ON TABLE dispatcher_exec_meta IS 'Execution metadata with SLA tracking and performance metrics';
COMMENT ON TABLE dispatcher_exec_step IS 'Step-level execution logs with detailed performance data';
COMMENT ON TABLE dispatcher_exec_result IS 'Execution results with business impact and quality metrics';

-- Verification query
SELECT 
    schemaname,
    tablename,
    tableowner,
    hasindexes,
    hasrules,
    hastriggers
FROM pg_tables 
WHERE tablename LIKE 'intent_registry_%' 
   OR tablename LIKE 'policy_gate_%'
   OR tablename LIKE 'capability_registry_%'
   OR tablename LIKE 'plan_synth_%'
   OR tablename LIKE 'dispatcher_exec_%'
ORDER BY tablename;
