-- Database Migration Scripts for RevAI Pro Platform
-- This file contains all database migration scripts for the platform

-- ============================================================================
-- MIGRATION 001: Create Core Schemas and Tables
-- ============================================================================

-- Create schemas
CREATE SCHEMA IF NOT EXISTS app_auth;
CREATE SCHEMA IF NOT EXISTS app_calendar;
CREATE SCHEMA IF NOT EXISTS app_letsmeet;
CREATE SCHEMA IF NOT EXISTS app_cruxx;
CREATE SCHEMA IF NOT EXISTS agent_automation;
CREATE SCHEMA IF NOT EXISTS ext_integrations;
CREATE SCHEMA IF NOT EXISTS app_audit;

-- ============================================================================
-- MIGRATION 002: Calendar Module Tables
-- ============================================================================

-- Calendar events table
CREATE TABLE IF NOT EXISTS app_calendar.events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    external_id VARCHAR(255),
    provider VARCHAR(50) NOT NULL, -- google, outlook, ical
    title VARCHAR(500) NOT NULL,
    description TEXT,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    location VARCHAR(500),
    attendees JSONB DEFAULT '[]'::jsonb,
    recurrence_rule JSONB,
    status VARCHAR(20) DEFAULT 'confirmed',
    visibility VARCHAR(20) DEFAULT 'private',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ,
    
    CONSTRAINT events_tenant_user_provider_external_unique 
        UNIQUE (tenant_id, user_id, provider, external_id)
);

-- Calendar free/busy cache
CREATE TABLE IF NOT EXISTS app_calendar.free_busy_cache (
    cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    date DATE NOT NULL,
    free_busy_data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT free_busy_cache_tenant_user_date_unique 
        UNIQUE (tenant_id, user_id, date)
);

-- Calendar overlays
CREATE TABLE IF NOT EXISTS app_calendar.overlays (
    overlay_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    event_id UUID REFERENCES app_calendar.events(event_id),
    overlay_type VARCHAR(50) NOT NULL, -- cruxx_reminder, meeting_badge, etc.
    overlay_data JSONB NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Calendar provider tokens
CREATE TABLE IF NOT EXISTS app_calendar.provider_tokens (
    token_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    provider VARCHAR(50) NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at TIMESTAMPTZ,
    scope TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT provider_tokens_tenant_user_provider_unique 
        UNIQUE (tenant_id, user_id, provider)
);

-- ============================================================================
-- MIGRATION 003: Let's Meet Module Tables
-- ============================================================================

-- Meetings table
CREATE TABLE IF NOT EXISTS app_letsmeet.meetings (
    meeting_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    calendar_event_id UUID REFERENCES app_calendar.events(event_id),
    audio_url TEXT,
    duration_seconds INTEGER,
    participants JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Transcripts table
CREATE TABLE IF NOT EXISTS app_letsmeet.transcripts (
    transcript_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES app_letsmeet.meetings(meeting_id),
    tenant_id UUID NOT NULL,
    language VARCHAR(10) DEFAULT 'en-US',
    transcript_text TEXT NOT NULL,
    segments JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Summaries table
CREATE TABLE IF NOT EXISTS app_letsmeet.summaries (
    summary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES app_letsmeet.meetings(meeting_id),
    tenant_id UUID NOT NULL,
    summary_text TEXT NOT NULL,
    key_points JSONB DEFAULT '[]'::jsonb,
    action_items JSONB DEFAULT '[]'::jsonb,
    decisions JSONB DEFAULT '[]'::jsonb,
    sentiment VARCHAR(20),
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings table (using pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS app_letsmeet.embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    meeting_id UUID REFERENCES app_letsmeet.meetings(meeting_id),
    segment_id UUID,
    text_content TEXT NOT NULL,
    embedding_vector vector(1536), -- OpenAI ada-002 dimensions
    embedding_model VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- MIGRATION 004: Cruxx Module Tables
-- ============================================================================

-- Cruxx actions table
CREATE TABLE IF NOT EXISTS app_cruxx.actions (
    cruxx_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    opportunity_id VARCHAR(255) NOT NULL, -- External CRM opportunity ID
    title VARCHAR(500) NOT NULL,
    description TEXT,
    assignee VARCHAR(255),
    due_date TIMESTAMPTZ NOT NULL,
    status VARCHAR(20) DEFAULT 'open', -- open, in_progress, completed, cancelled
    priority VARCHAR(20) DEFAULT 'medium', -- low, medium, high
    sla_status VARCHAR(20) DEFAULT 'met', -- met, warning, missed
    source_meeting_id UUID REFERENCES app_letsmeet.meetings(meeting_id),
    source_summary_id UUID REFERENCES app_letsmeet.summaries(summary_id),
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- ============================================================================
-- MIGRATION 005: Agent Automation Tables
-- ============================================================================

-- Agent registry
CREATE TABLE IF NOT EXISTS agent_automation.agents (
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    capabilities JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(20) DEFAULT 'active', -- active, inactive, retired, maintenance
    confidence_threshold DECIMAL(3,2) DEFAULT 0.8,
    trust_score DECIMAL(3,2) DEFAULT 0.5,
    region VARCHAR(50) NOT NULL,
    configuration JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMPTZ
);

-- Automation rules
CREATE TABLE IF NOT EXISTS agent_automation.automation_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    trigger VARCHAR(100) NOT NULL,
    conditions JSONB NOT NULL,
    actions JSONB NOT NULL,
    priority INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Run traces
CREATE TABLE IF NOT EXISTS agent_automation.run_traces (
    trace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    operation_type VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL, -- started, running, completed, failed
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    duration_ms INTEGER,
    overall_confidence_score DECIMAL(3,2),
    overall_trust_score DECIMAL(3,2),
    trust_drift DECIMAL(3,2) DEFAULT 0.0,
    compliance_flags JSONB DEFAULT '[]'::jsonb,
    audit_trail_id UUID,
    evidence_pack_id UUID,
    context JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Trace inputs
CREATE TABLE IF NOT EXISTS agent_automation.trace_inputs (
    input_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID REFERENCES agent_automation.run_traces(trace_id),
    input_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Trace decisions
CREATE TABLE IF NOT EXISTS agent_automation.trace_decisions (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID REFERENCES agent_automation.run_traces(trace_id),
    decision_type VARCHAR(50) NOT NULL, -- automated, user_override, system_fallback, human_review
    decision_point VARCHAR(100) NOT NULL,
    input_data JSONB NOT NULL,
    decision_criteria JSONB NOT NULL,
    decision_result JSONB NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    trust_score DECIMAL(3,2) NOT NULL,
    reasoning TEXT NOT NULL,
    alternatives_considered JSONB DEFAULT '[]'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Trace outputs
CREATE TABLE IF NOT EXISTS agent_automation.trace_outputs (
    output_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID REFERENCES agent_automation.run_traces(trace_id),
    output_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    performance_metrics JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- MIGRATION 006: Audit and Override Ledger Tables
-- ============================================================================

-- Override ledger
CREATE TABLE IF NOT EXISTS app_audit.override_ledger (
    override_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    override_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actor_type VARCHAR(20) NOT NULL, -- user, agent
    actor_id UUID NOT NULL,
    original_value JSONB,
    overridden_value JSONB,
    reason TEXT,
    context JSONB,
    evidence_link TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Event audit log
CREATE TABLE IF NOT EXISTS app_audit.event_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_id UUID NOT NULL,
    actor_id UUID NOT NULL,
    actor_type VARCHAR(20) NOT NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID NOT NULL,
    old_values JSONB,
    new_values JSONB,
    metadata JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- MIGRATION 007: External Integration Tables
-- ============================================================================

-- CRM integration mappings
CREATE TABLE IF NOT EXISTS ext_integrations.crm_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    crm_type VARCHAR(50) NOT NULL, -- salesforce, hubspot, etc.
    external_id VARCHAR(255) NOT NULL,
    internal_type VARCHAR(50) NOT NULL, -- opportunity, contact, etc.
    internal_id UUID NOT NULL,
    mapping_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT crm_mappings_tenant_crm_external_unique 
        UNIQUE (tenant_id, crm_type, external_id)
);

-- ============================================================================
-- MIGRATION 008: Indexes for Performance
-- ============================================================================

-- Calendar indexes
CREATE INDEX IF NOT EXISTS idx_events_tenant_user ON app_calendar.events (tenant_id, user_id);
CREATE INDEX IF NOT EXISTS idx_events_start_time ON app_calendar.events (start_time);
CREATE INDEX IF NOT EXISTS idx_events_provider ON app_calendar.events (provider);
CREATE INDEX IF NOT EXISTS idx_free_busy_cache_tenant_user_date ON app_calendar.free_busy_cache (tenant_id, user_id, date);
CREATE INDEX IF NOT EXISTS idx_overlays_tenant_user ON app_calendar.overlays (tenant_id, user_id);
CREATE INDEX IF NOT EXISTS idx_overlays_event_id ON app_calendar.overlays (event_id);

-- Let's Meet indexes
CREATE INDEX IF NOT EXISTS idx_meetings_tenant_user ON app_letsmeet.meetings (tenant_id, user_id);
CREATE INDEX IF NOT EXISTS idx_meetings_calendar_event ON app_letsmeet.meetings (calendar_event_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_meeting_id ON app_letsmeet.transcripts (meeting_id);
CREATE INDEX IF NOT EXISTS idx_summaries_meeting_id ON app_letsmeet.summaries (meeting_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_tenant_meeting ON app_letsmeet.embeddings (tenant_id, meeting_id);

-- Cruxx indexes
CREATE INDEX IF NOT EXISTS idx_cruxx_tenant ON app_cruxx.actions (tenant_id);
CREATE INDEX IF NOT EXISTS idx_cruxx_opportunity ON app_cruxx.actions (opportunity_id);
CREATE INDEX IF NOT EXISTS idx_cruxx_assignee ON app_cruxx.actions (assignee);
CREATE INDEX IF NOT EXISTS idx_cruxx_due_date ON app_cruxx.actions (due_date);
CREATE INDEX IF NOT EXISTS idx_cruxx_status ON app_cruxx.actions (status);

-- Agent automation indexes
CREATE INDEX IF NOT EXISTS idx_agents_tenant ON agent_automation.agents (tenant_id);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agent_automation.agents (status);
CREATE INDEX IF NOT EXISTS idx_automation_rules_tenant ON agent_automation.automation_rules (tenant_id);
CREATE INDEX IF NOT EXISTS idx_run_traces_tenant_user ON agent_automation.run_traces (tenant_id, user_id);
CREATE INDEX IF NOT EXISTS idx_run_traces_session ON agent_automation.run_traces (session_id);
CREATE INDEX IF NOT EXISTS idx_trace_inputs_trace_id ON agent_automation.trace_inputs (trace_id);
CREATE INDEX IF NOT EXISTS idx_trace_decisions_trace_id ON agent_automation.trace_decisions (trace_id);
CREATE INDEX IF NOT EXISTS idx_trace_outputs_trace_id ON agent_automation.trace_outputs (trace_id);

-- Audit indexes
CREATE INDEX IF NOT EXISTS idx_override_ledger_tenant ON app_audit.override_ledger (tenant_id);
CREATE INDEX IF NOT EXISTS idx_override_ledger_entity ON app_audit.override_ledger (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_override_ledger_actor ON app_audit.override_ledger (actor_type, actor_id);
CREATE INDEX IF NOT EXISTS idx_event_audit_tenant ON app_audit.event_audit_log (tenant_id);
CREATE INDEX IF NOT EXISTS idx_event_audit_event_type ON app_audit.event_audit_log (event_type);
CREATE INDEX IF NOT EXISTS idx_event_audit_timestamp ON app_audit.event_audit_log (timestamp);

-- External integration indexes
CREATE INDEX IF NOT EXISTS idx_crm_mappings_tenant ON ext_integrations.crm_mappings (tenant_id);
CREATE INDEX IF NOT EXISTS idx_crm_mappings_crm_type ON ext_integrations.crm_mappings (crm_type);

-- ============================================================================
-- MIGRATION 009: Row Level Security (RLS) Policies
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE app_calendar.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_calendar.free_busy_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_calendar.overlays ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_calendar.provider_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_letsmeet.meetings ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_letsmeet.transcripts ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_letsmeet.summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_letsmeet.embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_cruxx.actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_automation.agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_automation.automation_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_automation.run_traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_automation.trace_inputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_automation.trace_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_automation.trace_outputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_audit.override_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_audit.event_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE ext_integrations.crm_mappings ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (tenant isolation)
CREATE POLICY tenant_isolation_events ON app_calendar.events
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_free_busy_cache ON app_calendar.free_busy_cache
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_overlays ON app_calendar.overlays
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_provider_tokens ON app_calendar.provider_tokens
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_meetings ON app_letsmeet.meetings
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_transcripts ON app_letsmeet.transcripts
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_summaries ON app_letsmeet.summaries
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_embeddings ON app_letsmeet.embeddings
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_cruxx ON app_cruxx.actions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_agents ON agent_automation.agents
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_automation_rules ON agent_automation.automation_rules
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_run_traces ON agent_automation.run_traces
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_trace_inputs ON agent_automation.trace_inputs
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_trace_decisions ON agent_automation.trace_decisions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_trace_outputs ON agent_automation.trace_outputs
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_override_ledger ON app_audit.override_ledger
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_event_audit_log ON app_audit.event_audit_log
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_crm_mappings ON ext_integrations.crm_mappings
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ============================================================================
-- MIGRATION 010: Functions and Triggers
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_events_updated_at BEFORE UPDATE ON app_calendar.events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_overlays_updated_at BEFORE UPDATE ON app_calendar.overlays
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_provider_tokens_updated_at BEFORE UPDATE ON app_calendar.provider_tokens
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_meetings_updated_at BEFORE UPDATE ON app_letsmeet.meetings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cruxx_updated_at BEFORE UPDATE ON app_cruxx.actions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agent_automation.agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_automation_rules_updated_at BEFORE UPDATE ON agent_automation.automation_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_override_ledger_updated_at BEFORE UPDATE ON app_audit.override_ledger
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_crm_mappings_updated_at BEFORE UPDATE ON ext_integrations.crm_mappings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- MIGRATION 011: Initial Data and Configuration
-- ============================================================================

-- Insert default automation rules
INSERT INTO agent_automation.automation_rules (rule_id, tenant_id, trigger, conditions, actions, priority) VALUES
    (gen_random_uuid(), '00000000-0000-0000-0000-000000000000', 'calendar.event.created', 
     '{"event_type": "meeting", "auto_join_enabled": true}', 
     '["auto_join_meeting", "sync_overlays"]', 1),
    (gen_random_uuid(), '00000000-0000-0000-0000-000000000000', 'letsmeet.meeting.captured', 
     '{"auto_transcribe": true}', 
     '["auto_transcribe", "auto_summarize"]', 2),
    (gen_random_uuid(), '00000000-0000-0000-0000-000000000000', 'cruxx.action.created', 
     '{"auto_close_enabled": true}', 
     '["close_loops"]', 3)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- MIGRATION 012: Views for Common Queries
-- ============================================================================

-- View for active calendar events
CREATE OR REPLACE VIEW app_calendar.active_events AS
SELECT 
    e.*,
    CASE 
        WHEN e.end_time < CURRENT_TIMESTAMP THEN 'past'
        WHEN e.start_time <= CURRENT_TIMESTAMP AND e.end_time >= CURRENT_TIMESTAMP THEN 'current'
        ELSE 'future'
    END as event_status
FROM app_calendar.events e
WHERE e.deleted_at IS NULL;

-- View for meeting statistics
CREATE OR REPLACE VIEW app_letsmeet.meeting_stats AS
SELECT 
    m.tenant_id,
    COUNT(*) as total_meetings,
    COUNT(CASE WHEN m.status = 'completed' THEN 1 END) as completed_meetings,
    COUNT(CASE WHEN m.status = 'failed' THEN 1 END) as failed_meetings,
    AVG(m.duration_seconds) as avg_duration_seconds,
    COUNT(t.transcript_id) as meetings_with_transcripts,
    COUNT(s.summary_id) as meetings_with_summaries
FROM app_letsmeet.meetings m
LEFT JOIN app_letsmeet.transcripts t ON m.meeting_id = t.meeting_id
LEFT JOIN app_letsmeet.summaries s ON m.meeting_id = s.meeting_id
GROUP BY m.tenant_id;

-- View for Cruxx action statistics
CREATE OR REPLACE VIEW app_cruxx.action_stats AS
SELECT 
    tenant_id,
    COUNT(*) as total_actions,
    COUNT(CASE WHEN status = 'open' THEN 1 END) as open_actions,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_actions,
    COUNT(CASE WHEN sla_status = 'missed' THEN 1 END) as missed_sla_actions,
    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_completion_hours
FROM app_cruxx.actions
GROUP BY tenant_id;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Log migration completion
INSERT INTO app_audit.event_audit_log (
    audit_id, tenant_id, event_type, event_id, actor_id, actor_type, 
    action, resource_type, resource_id, new_values, timestamp
) VALUES (
    gen_random_uuid(), 
    '00000000-0000-0000-0000-000000000000', 
    'migration.completed', 
    gen_random_uuid(), 
    '00000000-0000-0000-0000-000000000000', 
    'system', 
    'create', 
    'database', 
    gen_random_uuid(), 
    '{"migration_version": "012", "description": "Complete database schema migration"}'::jsonb, 
    CURRENT_TIMESTAMP
);
