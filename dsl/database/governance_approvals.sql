-- Task 3.2.6: SoD & Approvals Database Schema
-- Separation of Duties and CAB approval workflows

-- Approval requests table
CREATE TABLE IF NOT EXISTS approval_requests (
    request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    industry VARCHAR(100) NOT NULL,
    developer_id VARCHAR(255) NOT NULL,
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    model_card_url TEXT,
    test_results JSONB,
    compliance_overlays TEXT[],
    justification TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'withdrawn')),
    approved_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    approved_at TIMESTAMP WITH TIME ZONE,
    sod_validated BOOLEAN DEFAULT FALSE,
    
    -- RLS (Row Level Security) for tenant isolation
    CONSTRAINT tenant_isolation CHECK (tenant_id IS NOT NULL)
);

-- CAB approvals table
CREATE TABLE IF NOT EXISTS cab_approvals (
    approval_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL REFERENCES approval_requests(request_id),
    cab_member_id VARCHAR(255) NOT NULL,
    cab_member_role VARCHAR(255) NOT NULL,
    decision VARCHAR(20) NOT NULL CHECK (decision IN ('pending', 'approved', 'rejected', 'withdrawn')),
    decision_rationale TEXT NOT NULL,
    conditions TEXT[],
    evidence_hash VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure CAB member is not the developer (SoD enforcement)
    CONSTRAINT sod_check CHECK (cab_member_id != (
        SELECT developer_id FROM approval_requests WHERE request_id = cab_approvals.request_id
    ))
);

-- Tenant risk owners table
CREATE TABLE IF NOT EXISTS tenant_risk_owners (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    risk_owner_id VARCHAR(255) NOT NULL,
    risk_owner_name VARCHAR(255) NOT NULL,
    risk_owner_role VARCHAR(255) NOT NULL,
    approval_authority TEXT[] NOT NULL, -- Array of risk levels they can approve
    active BOOLEAN DEFAULT TRUE,
    appointed_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint to prevent duplicate risk owners per tenant
    UNIQUE(tenant_id, risk_owner_id)
);

-- Evidence store table (immutable, append-only)
CREATE TABLE IF NOT EXISTS approval_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL REFERENCES approval_requests(request_id),
    evidence_type VARCHAR(100) NOT NULL, -- 'request', 'sod_check', 'approval', 'audit'
    evidence_data JSONB NOT NULL,
    evidence_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Immutable - no updates allowed
    created_by VARCHAR(255) NOT NULL
);

-- Audit log for all approval activities
CREATE TABLE IF NOT EXISTS approval_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID REFERENCES approval_requests(request_id),
    action VARCHAR(100) NOT NULL, -- 'created', 'approved', 'rejected', 'sod_violation', etc.
    actor_id VARCHAR(255) NOT NULL,
    actor_role VARCHAR(255),
    details JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tenant_id VARCHAR(255) NOT NULL
);

-- Enable Row Level Security for tenant isolation
ALTER TABLE approval_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE cab_approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_risk_owners ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_audit_log ENABLE ROW LEVEL SECURITY;

-- RLS Policies (placeholder - actual implementation would use proper tenant context)
CREATE POLICY tenant_isolation_approval_requests ON approval_requests
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_cab_approvals ON cab_approvals
    FOR ALL TO rbia_users
    USING (EXISTS (
        SELECT 1 FROM approval_requests ar 
        WHERE ar.request_id = cab_approvals.request_id 
        AND ar.tenant_id = current_setting('rbia.current_tenant_id', true)
    ));

CREATE POLICY tenant_isolation_risk_owners ON tenant_risk_owners
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_evidence ON approval_evidence
    FOR ALL TO rbia_users
    USING (EXISTS (
        SELECT 1 FROM approval_requests ar 
        WHERE ar.request_id = approval_evidence.request_id 
        AND ar.tenant_id = current_setting('rbia.current_tenant_id', true)
    ));

CREATE POLICY tenant_isolation_audit ON approval_audit_log
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

-- Indexes for performance
CREATE INDEX idx_approval_requests_tenant_id ON approval_requests(tenant_id);
CREATE INDEX idx_approval_requests_developer_id ON approval_requests(developer_id);
CREATE INDEX idx_approval_requests_status ON approval_requests(status);
CREATE INDEX idx_cab_approvals_request_id ON cab_approvals(request_id);
CREATE INDEX idx_cab_approvals_member_id ON cab_approvals(cab_member_id);
CREATE INDEX idx_tenant_risk_owners_tenant_id ON tenant_risk_owners(tenant_id);
CREATE INDEX idx_approval_evidence_request_id ON approval_evidence(request_id);
CREATE INDEX idx_approval_audit_log_tenant_id ON approval_audit_log(tenant_id);
CREATE INDEX idx_approval_audit_log_timestamp ON approval_audit_log(timestamp);

-- Functions for SoD validation
CREATE OR REPLACE FUNCTION validate_sod_approval()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if approver is the same as developer
    IF NEW.cab_member_id = (
        SELECT developer_id FROM approval_requests WHERE request_id = NEW.request_id
    ) THEN
        RAISE EXCEPTION 'SoD Violation: Developer cannot approve their own model (Request: %, Developer: %, Approver: %)', 
            NEW.request_id, 
            (SELECT developer_id FROM approval_requests WHERE request_id = NEW.request_id),
            NEW.cab_member_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to enforce SoD on CAB approvals
CREATE TRIGGER enforce_sod_on_approval
    BEFORE INSERT OR UPDATE ON cab_approvals
    FOR EACH ROW
    EXECUTE FUNCTION validate_sod_approval();

-- Function to log audit events
CREATE OR REPLACE FUNCTION log_approval_audit()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO approval_audit_log (request_id, action, actor_id, details, tenant_id)
    VALUES (
        COALESCE(NEW.request_id, OLD.request_id),
        TG_OP,
        COALESCE(NEW.cab_member_id, NEW.developer_id, 'system'),
        jsonb_build_object(
            'old', to_jsonb(OLD),
            'new', to_jsonb(NEW),
            'operation', TG_OP,
            'table', TG_TABLE_NAME
        ),
        COALESCE(NEW.tenant_id, OLD.tenant_id, (
            SELECT tenant_id FROM approval_requests 
            WHERE request_id = COALESCE(NEW.request_id, OLD.request_id)
        ))
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Audit triggers
CREATE TRIGGER audit_approval_requests
    AFTER INSERT OR UPDATE OR DELETE ON approval_requests
    FOR EACH ROW
    EXECUTE FUNCTION log_approval_audit();

CREATE TRIGGER audit_cab_approvals
    AFTER INSERT OR UPDATE OR DELETE ON cab_approvals
    FOR EACH ROW
    EXECUTE FUNCTION log_approval_audit();

-- Sample data for testing (remove in production)
INSERT INTO tenant_risk_owners (tenant_id, risk_owner_id, risk_owner_name, risk_owner_role, approval_authority) VALUES
('tenant_saas_001', 'user_cro_001', 'John Smith', 'Chief Risk Officer', ARRAY['low', 'medium', 'high']),
('tenant_saas_001', 'user_compliance_001', 'Jane Doe', 'Compliance Manager', ARRAY['low', 'medium']),
('tenant_banking_001', 'user_cro_002', 'Bob Johnson', 'Chief Risk Officer', ARRAY['low', 'medium', 'high', 'critical']),
('tenant_banking_001', 'user_compliance_002', 'Alice Brown', 'Compliance Officer', ARRAY['low', 'medium', 'high']);

COMMENT ON TABLE approval_requests IS 'Task 3.2.6: Model approval requests with SoD validation';
COMMENT ON TABLE cab_approvals IS 'Task 3.2.6: CAB (Change Advisory Board) approval decisions';
COMMENT ON TABLE tenant_risk_owners IS 'Task 3.2.6: Per-tenant risk owners with approval authority';
COMMENT ON TABLE approval_evidence IS 'Task 3.2.6: Immutable evidence store for approvals';
COMMENT ON TABLE approval_audit_log IS 'Task 3.2.6: Audit log for all approval activities';
