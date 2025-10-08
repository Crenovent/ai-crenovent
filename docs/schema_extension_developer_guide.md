# Schema Extension Developer Guide

**Task 7.2-T58: Write developer guide (how to extend schema safely)**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Engineering Team

---

## Overview

This guide provides developers with best practices, patterns, and procedures for safely extending the RevOps database schema. Following these guidelines ensures schema changes are backward compatible, performant, and maintain data integrity.

---

## Table of Contents

1. [Before You Start](#before-you-start)
2. [Schema Extension Principles](#schema-extension-principles)
3. [Safe Extension Patterns](#safe-extension-patterns)
4. [Migration Best Practices](#migration-best-practices)
5. [Testing Schema Changes](#testing-schema-changes)
6. [Performance Considerations](#performance-considerations)
7. [Common Pitfalls](#common-pitfalls)
8. [Review Process](#review-process)

---

## Before You Start

### Prerequisites
- Read the [Business Glossary](business_glossary.md) for standard terminology
- Review [Schema Conformance Rules](schema_conformance_rules.md) for compliance requirements
- Understand the existing schema structure and relationships
- Have access to development and staging environments

### Required Tools
- PostgreSQL client (psql, pgAdmin, or similar)
- Migration generator: `ai-crenovent/dsl/schemas/migration_generator.py`
- DDL generator: `ai-crenovent/dsl/schemas/ddl_generator.py`
- Schema validation tools

### Planning Checklist
- [ ] Business requirement clearly defined
- [ ] Impact assessment completed
- [ ] Backward compatibility verified
- [ ] Performance impact evaluated
- [ ] Migration strategy planned
- [ ] Rollback plan prepared
- [ ] Testing strategy defined

---

## Schema Extension Principles

### 1. Backward Compatibility First
**Always maintain backward compatibility** - existing applications should continue to work without modification.

```sql
-- ✅ GOOD: Adding optional column
ALTER TABLE accounts ADD COLUMN industry_segment VARCHAR(50);

-- ❌ BAD: Changing existing column type
ALTER TABLE accounts ALTER COLUMN annual_revenue TYPE BIGINT;
```

### 2. Additive Changes Only
**Prefer additive changes** over modifications to existing structures.

```sql
-- ✅ GOOD: Add new table for additional data
CREATE TABLE account_segments (
    account_id UUID REFERENCES accounts(account_id),
    segment_type VARCHAR(50) NOT NULL,
    segment_value VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ❌ BAD: Modifying existing table structure significantly
ALTER TABLE accounts DROP COLUMN account_type;
```

### 3. Fail-Safe Defaults
**Use safe defaults** for new columns to avoid breaking existing data.

```sql
-- ✅ GOOD: Safe default value
ALTER TABLE opportunities 
ADD COLUMN risk_score DECIMAL(3,2) DEFAULT 0.50 CHECK (risk_score BETWEEN 0 AND 1);

-- ❌ BAD: No default for NOT NULL column
ALTER TABLE opportunities 
ADD COLUMN required_field VARCHAR(50) NOT NULL;
```

### 4. Incremental Rollout
**Plan for incremental rollout** with feature flags and gradual adoption.

```sql
-- ✅ GOOD: Optional feature with flag
ALTER TABLE workflows 
ADD COLUMN enhanced_mode BOOLEAN DEFAULT FALSE;

-- Application can gradually enable enhanced_mode per tenant
```

---

## Safe Extension Patterns

### Pattern 1: Adding Optional Columns

**Use Case:** Adding new optional fields to existing entities.

```sql
-- Step 1: Add column with safe default
ALTER TABLE accounts 
ADD COLUMN customer_health_score DECIMAL(3,2) DEFAULT 0.75 
CHECK (customer_health_score BETWEEN 0 AND 1);

-- Step 2: Add index if needed for queries
CREATE INDEX CONCURRENTLY idx_accounts_health_score 
ON accounts(customer_health_score) 
WHERE customer_health_score IS NOT NULL;

-- Step 3: Update application to populate new field
-- (Application deployment)

-- Step 4: Backfill existing records (optional)
UPDATE accounts 
SET customer_health_score = calculate_health_score(account_id) 
WHERE customer_health_score = 0.75; -- Only update defaults
```

### Pattern 2: Creating Extension Tables

**Use Case:** Adding complex new functionality without modifying core tables.

```sql
-- Create extension table with proper relationships
CREATE TABLE account_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    
    -- Analytics fields
    engagement_score DECIMAL(3,2) DEFAULT 0.50,
    growth_trend VARCHAR(20) DEFAULT 'stable',
    risk_indicators JSONB DEFAULT '[]',
    
    -- Standard audit fields
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT uk_account_analytics_account UNIQUE (account_id),
    CONSTRAINT chk_engagement_score CHECK (engagement_score BETWEEN 0 AND 1),
    CONSTRAINT chk_growth_trend CHECK (growth_trend IN ('declining', 'stable', 'growing', 'accelerating'))
);

-- Add RLS policy for tenant isolation
ALTER TABLE account_analytics ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_account_analytics ON account_analytics
FOR ALL USING (tenant_id = current_setting('app.tenant_id')::integer);

-- Add indexes
CREATE INDEX idx_account_analytics_tenant ON account_analytics(tenant_id);
CREATE INDEX idx_account_analytics_engagement ON account_analytics(engagement_score);
```

### Pattern 3: Versioned Schema Evolution

**Use Case:** Major schema changes that need gradual migration.

```sql
-- Step 1: Create new version of table
CREATE TABLE opportunities_v2 (
    opportunity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    account_id UUID NOT NULL REFERENCES accounts(account_id),
    
    -- Enhanced fields
    opportunity_name VARCHAR(255) NOT NULL,
    stage_id UUID REFERENCES opportunity_stages(stage_id), -- New normalized stages
    amount DECIMAL(15,2) NOT NULL CHECK (amount >= 0),
    probability DECIMAL(5,2) NOT NULL CHECK (probability BETWEEN 0 AND 100),
    
    -- New fields
    weighted_amount DECIMAL(15,2) GENERATED ALWAYS AS (amount * probability / 100) STORED,
    stage_duration_days INTEGER DEFAULT 0,
    last_activity_date DATE,
    
    -- Standard fields
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    is_deleted BOOLEAN DEFAULT FALSE
);

-- Step 2: Create migration view for backward compatibility
CREATE VIEW opportunities AS
SELECT 
    opportunity_id,
    tenant_id,
    account_id,
    opportunity_name,
    s.stage_name as stage, -- Map back to old format
    amount,
    probability,
    created_at,
    updated_at,
    is_deleted
FROM opportunities_v2 o
LEFT JOIN opportunity_stages s ON o.stage_id = s.stage_id
WHERE NOT is_deleted;

-- Step 3: Create INSTEAD OF triggers for backward compatibility
CREATE OR REPLACE FUNCTION opportunities_insert_trigger()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO opportunities_v2 (
        opportunity_id, tenant_id, account_id, opportunity_name,
        stage_id, amount, probability
    ) VALUES (
        COALESCE(NEW.opportunity_id, gen_random_uuid()),
        NEW.tenant_id,
        NEW.account_id,
        NEW.opportunity_name,
        (SELECT stage_id FROM opportunity_stages WHERE stage_name = NEW.stage LIMIT 1),
        NEW.amount,
        NEW.probability
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER opportunities_insert 
INSTEAD OF INSERT ON opportunities
FOR EACH ROW EXECUTE FUNCTION opportunities_insert_trigger();
```

### Pattern 4: Adding Computed Columns

**Use Case:** Adding derived fields that are calculated from existing data.

```sql
-- Add computed column using GENERATED ALWAYS
ALTER TABLE opportunities 
ADD COLUMN weighted_amount DECIMAL(15,2) 
GENERATED ALWAYS AS (amount * probability / 100) STORED;

-- Add index on computed column
CREATE INDEX CONCURRENTLY idx_opportunities_weighted_amount 
ON opportunities(weighted_amount) 
WHERE weighted_amount > 0;
```

### Pattern 5: JSON Schema Evolution

**Use Case:** Extending JSONB fields with new structure while maintaining compatibility.

```sql
-- Existing JSONB field: metadata
-- Old format: {"source": "salesforce", "sync_date": "2024-01-01"}
-- New format: {"source": "salesforce", "sync_date": "2024-01-01", "sync_details": {...}}

-- Add validation function
CREATE OR REPLACE FUNCTION validate_account_metadata(metadata JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Validate required fields exist
    IF NOT (metadata ? 'source') THEN
        RETURN FALSE;
    END IF;
    
    -- Validate sync_details structure if present
    IF metadata ? 'sync_details' THEN
        IF NOT jsonb_typeof(metadata->'sync_details') = 'object' THEN
            RETURN FALSE;
        END IF;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Add check constraint for validation
ALTER TABLE accounts 
ADD CONSTRAINT chk_account_metadata_valid 
CHECK (validate_account_metadata(metadata));
```

---

## Migration Best Practices

### 1. Use Migration Generator

Always use the provided migration generator for consistency:

```python
from ai_crenovent.dsl.schemas.migration_generator import MigrationGenerator

# Create migration
generator = MigrationGenerator()
migration = generator.create_migration(
    title="Add Customer Health Score to Accounts",
    description="Add customer_health_score field for account analytics",
    author="Your Name"
)

# Add migration steps
generator.add_add_column_step(
    migration,
    table_name="accounts",
    column_name="customer_health_score",
    column_definition="DECIMAL(3,2) DEFAULT 0.75 CHECK (customer_health_score BETWEEN 0 AND 1)"
)

# Save migration file
generator.save_migration(migration)
```

### 2. Migration Naming Convention

Follow consistent naming for migration files:
- Format: `YYYYMMDD_HHMMSS_descriptive_name.sql`
- Example: `20241008_143000_add_customer_health_score_to_accounts.sql`

### 3. Migration Structure

Structure migrations with clear sections:

```sql
-- =====================================================
-- MIGRATION: Add Customer Health Score to Accounts
-- Version: v20241008_143000
-- Author: Platform Team
-- Risk Assessment: low
-- Estimated Duration: < 1 minute
-- =====================================================

-- Description: Add customer_health_score field for account analytics

-- =====================================================
-- UP MIGRATION
-- =====================================================

-- Step 1: Add customer_health_score column
ALTER TABLE accounts 
ADD COLUMN customer_health_score DECIMAL(3,2) DEFAULT 0.75 
CHECK (customer_health_score BETWEEN 0 AND 1);

-- Verify step completion
SELECT COUNT(*) FROM information_schema.columns 
WHERE table_name = 'accounts' AND column_name = 'customer_health_score';

-- Step 2: Add index for performance
CREATE INDEX CONCURRENTLY idx_accounts_health_score 
ON accounts(customer_health_score) 
WHERE customer_health_score != 0.75;

-- =====================================================
-- DOWN MIGRATION (ROLLBACK)
-- =====================================================

/*
-- Rollback Step 2: Drop index
DROP INDEX IF EXISTS idx_accounts_health_score;

-- Rollback Step 1: Drop column
ALTER TABLE accounts DROP COLUMN IF EXISTS customer_health_score;
*/
```

### 4. Testing Migrations

Test migrations thoroughly before production:

```bash
# 1. Test on development environment
psql -d dev_database -f migration_file.sql

# 2. Test rollback
psql -d dev_database -c "/* Execute rollback steps */"

# 3. Test on staging with production-like data
psql -d staging_database -f migration_file.sql

# 4. Performance test with large datasets
EXPLAIN ANALYZE /* migration queries */
```

---

## Testing Schema Changes

### 1. Unit Tests for Schema

Create tests for schema validation:

```python
# test_schema_extensions.py
import pytest
from sqlalchemy import create_engine, text

def test_customer_health_score_constraint():
    """Test that customer_health_score constraint works correctly"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Test valid value
        conn.execute(text("""
            INSERT INTO accounts (account_id, tenant_id, account_name, customer_health_score)
            VALUES (gen_random_uuid(), 1000, 'Test Account', 0.85)
        """))
        
        # Test invalid value (should fail)
        with pytest.raises(Exception):
            conn.execute(text("""
                INSERT INTO accounts (account_id, tenant_id, account_name, customer_health_score)
                VALUES (gen_random_uuid(), 1000, 'Test Account 2', 1.5)
            """))
```

### 2. Integration Tests

Test schema changes with application code:

```python
def test_account_with_health_score():
    """Test that application can work with new health score field"""
    account = Account.create(
        tenant_id=1000,
        account_name="Test Account",
        customer_health_score=0.85
    )
    
    assert account.customer_health_score == 0.85
    assert account.is_healthy()  # New method using health score
```

### 3. Performance Tests

Test performance impact of schema changes:

```sql
-- Test query performance with new index
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM accounts 
WHERE customer_health_score > 0.8 
AND tenant_id = 1000;

-- Test insert performance with new constraints
EXPLAIN (ANALYZE, BUFFERS)
INSERT INTO accounts (account_id, tenant_id, account_name, customer_health_score)
SELECT gen_random_uuid(), 1000, 'Test ' || generate_series, 0.75
FROM generate_series(1, 1000);
```

---

## Performance Considerations

### 1. Index Strategy

Add indexes carefully to balance query performance and write overhead:

```sql
-- ✅ GOOD: Selective index with WHERE clause
CREATE INDEX CONCURRENTLY idx_accounts_high_health_score 
ON accounts(customer_health_score) 
WHERE customer_health_score > 0.8;

-- ❌ BAD: Full index on low-selectivity column
CREATE INDEX idx_accounts_all_health_scores 
ON accounts(customer_health_score);
```

### 2. Constraint Performance

Use efficient constraints that don't slow down writes:

```sql
-- ✅ GOOD: Simple range check
ALTER TABLE accounts 
ADD CONSTRAINT chk_health_score_range 
CHECK (customer_health_score BETWEEN 0 AND 1);

-- ❌ BAD: Complex constraint with subquery
ALTER TABLE accounts 
ADD CONSTRAINT chk_health_score_complex 
CHECK (customer_health_score > (SELECT AVG(customer_health_score) FROM accounts) - 0.5);
```

### 3. Large Table Modifications

For large tables, use strategies to minimize downtime:

```sql
-- Strategy 1: Add column with default, then update in batches
ALTER TABLE large_table ADD COLUMN new_field INTEGER DEFAULT 0;

-- Update in batches to avoid long locks
DO $$
DECLARE
    batch_size INTEGER := 10000;
    total_updated INTEGER := 0;
BEGIN
    LOOP
        UPDATE large_table 
        SET new_field = calculate_new_value(id)
        WHERE new_field = 0 
        AND id IN (
            SELECT id FROM large_table 
            WHERE new_field = 0 
            LIMIT batch_size
        );
        
        GET DIAGNOSTICS total_updated = ROW_COUNT;
        EXIT WHEN total_updated = 0;
        
        -- Small delay to allow other operations
        PERFORM pg_sleep(0.1);
    END LOOP;
END $$;
```

---

## Common Pitfalls

### 1. Breaking Changes

**Avoid these breaking changes:**

```sql
-- ❌ DON'T: Drop columns (breaks existing queries)
ALTER TABLE accounts DROP COLUMN old_field;

-- ❌ DON'T: Rename columns (breaks existing code)
ALTER TABLE accounts RENAME COLUMN account_name TO name;

-- ❌ DON'T: Change column types (can cause data loss)
ALTER TABLE accounts ALTER COLUMN annual_revenue TYPE INTEGER;

-- ❌ DON'T: Add NOT NULL without default (breaks inserts)
ALTER TABLE accounts ADD COLUMN required_field VARCHAR(50) NOT NULL;
```

### 2. Performance Killers

**Avoid these performance issues:**

```sql
-- ❌ DON'T: Create indexes without CONCURRENTLY (locks table)
CREATE INDEX idx_accounts_name ON accounts(account_name);

-- ❌ DON'T: Add foreign keys without validation (long lock)
ALTER TABLE opportunities ADD CONSTRAINT fk_opp_account 
FOREIGN KEY (account_id) REFERENCES accounts(account_id);

-- ✅ DO: Use NOT VALID first, then validate separately
ALTER TABLE opportunities ADD CONSTRAINT fk_opp_account 
FOREIGN KEY (account_id) REFERENCES accounts(account_id) NOT VALID;

-- Later, in separate transaction:
ALTER TABLE opportunities VALIDATE CONSTRAINT fk_opp_account;
```

### 3. Data Integrity Issues

**Prevent these data problems:**

```sql
-- ❌ DON'T: Forget tenant isolation on new tables
CREATE TABLE new_table (
    id UUID PRIMARY KEY,
    data VARCHAR(255)
    -- Missing tenant_id and RLS policy
);

-- ✅ DO: Always include tenant isolation
CREATE TABLE new_table (
    id UUID PRIMARY KEY,
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    data VARCHAR(255)
);

ALTER TABLE new_table ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_new_table ON new_table
FOR ALL USING (tenant_id = current_setting('app.tenant_id')::integer);
```

---

## Review Process

### 1. Pre-Review Checklist

Before submitting schema changes for review:

- [ ] Migration script created and tested
- [ ] Rollback plan documented and tested
- [ ] Performance impact assessed
- [ ] Conformance rules followed
- [ ] Unit tests written
- [ ] Documentation updated
- [ ] Backward compatibility verified

### 2. Review Criteria

Schema changes will be reviewed for:

- **Correctness:** Does the change solve the business requirement?
- **Safety:** Is the change backward compatible and safe to deploy?
- **Performance:** What is the performance impact?
- **Maintainability:** Is the change consistent with existing patterns?
- **Documentation:** Is the change properly documented?

### 3. Approval Process

1. **Developer Review:** Peer review by another developer
2. **Architecture Review:** Review by platform architecture team
3. **DBA Review:** Review by database administrator (for complex changes)
4. **Security Review:** Review by security team (for sensitive data)
5. **Final Approval:** Approval by platform engineering lead

### 4. Deployment Process

1. **Staging Deployment:** Deploy to staging environment first
2. **Validation:** Validate changes work as expected
3. **Production Deployment:** Deploy during maintenance window
4. **Monitoring:** Monitor for issues after deployment
5. **Rollback Plan:** Be ready to rollback if issues occur

---

## Tools and Resources

### Schema Management Tools
- **Migration Generator:** `ai-crenovent/dsl/schemas/migration_generator.py`
- **DDL Generator:** `ai-crenovent/dsl/schemas/ddl_generator.py`
- **Sample Data Generator:** `ai-crenovent/test_data/sample_datasets_generator.py`

### Documentation
- **Business Glossary:** `ai-crenovent/docs/business_glossary.md`
- **Conformance Rules:** `ai-crenovent/docs/schema_conformance_rules.md`
- **Architecture Docs:** `ai-crenovent/docs/architecture/`

### Testing Resources
- **Sample Datasets:** `ai-crenovent/test_data/samples/`
- **Test Database:** Available in development environment
- **Performance Testing:** Use staging environment with production data size

---

## Getting Help

### Contact Information
- **Platform Engineering Team:** platform-engineering@company.com
- **Database Team:** database-team@company.com
- **Architecture Team:** architecture@company.com

### Office Hours
- **Schema Review Sessions:** Tuesdays 2-3 PM
- **Architecture Office Hours:** Thursdays 10-11 AM
- **Database Office Hours:** Fridays 3-4 PM

### Resources
- **Internal Wiki:** [Schema Extension Guidelines](wiki-link)
- **Slack Channels:** #platform-engineering, #database-help
- **Training Materials:** Available in learning management system

---

## Conclusion

Safe schema extension is critical for maintaining a stable, performant platform. By following these guidelines, you can ensure your schema changes are:

- **Backward Compatible:** Existing applications continue to work
- **Performant:** Changes don't negatively impact system performance
- **Maintainable:** Changes follow consistent patterns and are well-documented
- **Compliant:** Changes meet all governance and conformance requirements

Remember: **When in doubt, ask for help!** The platform engineering team is here to support you in making safe, effective schema changes.

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Feedback:** Please provide feedback on this guide to platform-engineering@company.com
