# Canonical RevOps Ontology

**Task 7.2-T01: Publish canonical ontology (entities, attributes, relationships)**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Data Architecture Team

---

## Overview

This document defines the canonical RevOps ontology that serves as the foundation for all database schema mapping, data integration, and business intelligence across the RBA platform. The ontology provides a shared domain language for Revenue Operations across multiple industries.

---

## Core Domain Entities

### **1. Tenant & Identity Management**

#### **Tenant**
```yaml
entity: tenant
description: "Multi-tenant isolation boundary"
attributes:
  - tenant_id: "Primary identifier (integer)"
  - tenant_name: "Display name"
  - tenant_code: "Short code (e.g., 'ACME')"
  - industry_code: "Industry classification (SaaS, BANK, INSUR)"
  - region: "Primary region (US, EU, IN, APAC)"
  - status: "active, suspended, archived"
  - created_at: "Tenant creation timestamp"
  - subscription_tier: "free, standard, premium, enterprise"
  - compliance_frameworks: "Array of applicable frameworks"
relationships:
  - has_many: users, accounts, workflows, policies
  - belongs_to: parent_tenant (for sub-tenants)
```

#### **User**
```yaml
entity: user
description: "System users with role-based access"
attributes:
  - user_id: "Primary identifier (integer)"
  - tenant_id: "Tenant isolation key"
  - email: "Unique email address"
  - first_name: "Given name"
  - last_name: "Family name"
  - role: "System role (admin, user, viewer)"
  - status: "active, inactive, suspended"
  - last_login_at: "Last authentication timestamp"
  - profile: "JSONB profile data"
  - reports_to: "Manager user_id"
relationships:
  - belongs_to: tenant
  - has_many: activities, workflows_created, approvals
  - reports_to: user (self-referential)
```

#### **Role**
```yaml
entity: role
description: "Role-based access control definitions"
attributes:
  - role_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - role_name: "Role display name"
  - permissions: "Array of permission strings"
  - geographic_scope: "Territory/region limitations"
  - industry_scope: "Industry-specific permissions"
relationships:
  - belongs_to: tenant
  - has_many: user_roles
```

---

### **2. Customer Relationship Management (CRM)**

#### **Account**
```yaml
entity: account
description: "Customer/prospect organizations"
attributes:
  - account_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_name: "Organization name"
  - account_type: "prospect, customer, partner, competitor"
  - industry: "Industry classification"
  - annual_revenue: "Annual revenue (decimal)"
  - employee_count: "Number of employees"
  - website: "Company website URL"
  - billing_address: "JSONB address structure"
  - shipping_address: "JSONB address structure"
  - account_tier: "enterprise, mid_market, smb"
  - health_score: "Customer health (0.0-1.0)"
  - churn_risk: "low, medium, high, critical"
  - external_id: "CRM system identifier"
  - source_system: "salesforce, hubspot, pipedrive"
relationships:
  - belongs_to: tenant
  - has_many: contacts, opportunities, contracts, cases
  - has_one: account_owner (user)
```

#### **Contact**
```yaml
entity: contact
description: "Individual people at accounts"
attributes:
  - contact_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_id: "Parent account"
  - first_name: "Given name"
  - last_name: "Family name"
  - email: "Email address"
  - phone: "Phone number"
  - title: "Job title"
  - department: "Department/function"
  - decision_maker: "Boolean decision maker flag"
  - influence_level: "low, medium, high"
  - contact_status: "active, inactive, bounced"
  - external_id: "CRM system identifier"
relationships:
  - belongs_to: tenant, account
  - has_many: activities, opportunities (as contact_role)
```

#### **Lead**
```yaml
entity: lead
description: "Unqualified prospects"
attributes:
  - lead_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - first_name: "Given name"
  - last_name: "Family name"
  - email: "Email address"
  - company: "Company name"
  - title: "Job title"
  - lead_source: "website, referral, campaign, event"
  - lead_status: "new, contacted, qualified, unqualified"
  - lead_score: "Numeric scoring (0-100)"
  - qualification_date: "When qualified/disqualified"
  - converted_account_id: "If converted to account"
  - converted_contact_id: "If converted to contact"
  - converted_opportunity_id: "If converted to opportunity"
relationships:
  - belongs_to: tenant
  - converts_to: account, contact, opportunity
```

---

### **3. Sales & Pipeline Management**

#### **Opportunity**
```yaml
entity: opportunity
description: "Sales deals/opportunities"
attributes:
  - opportunity_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_id: "Parent account"
  - opportunity_name: "Deal name"
  - stage: "Current sales stage"
  - amount: "Deal value (decimal)"
  - currency: "Currency code (USD, EUR, INR)"
  - probability: "Win probability (0-100)"
  - close_date: "Expected close date"
  - actual_close_date: "Actual close date"
  - lead_source: "How opportunity originated"
  - deal_type: "new_business, expansion, renewal"
  - forecast_category: "commit, best_case, pipeline, omitted"
  - next_step: "Next action required"
  - competition: "Competitive situation"
  - loss_reason: "If lost, reason"
  - external_id: "CRM system identifier"
relationships:
  - belongs_to: tenant, account
  - has_one: opportunity_owner (user)
  - has_many: opportunity_products, activities
```

#### **Stage**
```yaml
entity: stage
description: "Sales process stages"
attributes:
  - stage_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - stage_name: "Stage display name"
  - stage_order: "Sequence order (1, 2, 3...)"
  - probability: "Default probability for stage"
  - stage_type: "prospecting, qualification, proposal, negotiation, closed"
  - is_closed: "Boolean closed stage flag"
  - is_won: "Boolean won stage flag"
relationships:
  - belongs_to: tenant
  - has_many: opportunities
```

#### **Pipeline**
```yaml
entity: pipeline
description: "Sales pipeline definitions"
attributes:
  - pipeline_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - pipeline_name: "Pipeline display name"
  - pipeline_type: "sales, marketing, support"
  - is_default: "Boolean default pipeline flag"
  - stages: "Array of stage configurations"
relationships:
  - belongs_to: tenant
  - has_many: opportunities, stages
```

---

### **4. Product & Pricing**

#### **Product**
```yaml
entity: product
description: "Products and services offered"
attributes:
  - product_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - product_name: "Product display name"
  - product_code: "SKU/product code"
  - product_family: "Product family grouping"
  - product_type: "software, service, hardware"
  - description: "Product description"
  - is_active: "Boolean active flag"
  - unit_price: "Base unit price"
  - currency: "Currency code"
  - billing_frequency: "monthly, quarterly, annually"
  - product_category: "core, addon, professional_services"
relationships:
  - belongs_to: tenant
  - has_many: opportunity_products, contract_line_items
```

#### **PriceBook**
```yaml
entity: pricebook
description: "Pricing configurations"
attributes:
  - pricebook_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - pricebook_name: "Pricebook display name"
  - is_standard: "Boolean standard pricebook flag"
  - currency: "Currency code"
  - effective_date: "When pricing becomes effective"
  - expiration_date: "When pricing expires"
relationships:
  - belongs_to: tenant
  - has_many: pricebook_entries
```

#### **Discount**
```yaml
entity: discount
description: "Discount rules and applications"
attributes:
  - discount_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - discount_name: "Discount display name"
  - discount_type: "percentage, fixed_amount, volume"
  - discount_value: "Discount amount/percentage"
  - minimum_quantity: "Minimum quantity for discount"
  - maximum_discount: "Maximum discount allowed"
  - valid_from: "Discount validity start"
  - valid_to: "Discount validity end"
relationships:
  - belongs_to: tenant
  - applies_to: products, opportunities
```

---

### **5. Contracts & Subscriptions**

#### **Contract**
```yaml
entity: contract
description: "Legal agreements with customers"
attributes:
  - contract_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_id: "Customer account"
  - contract_number: "Human-readable contract number"
  - contract_type: "master, amendment, renewal"
  - status: "draft, active, expired, terminated"
  - start_date: "Contract start date"
  - end_date: "Contract end date"
  - auto_renew: "Boolean auto-renewal flag"
  - renewal_term: "Renewal term length"
  - total_value: "Total contract value"
  - currency: "Currency code"
  - billing_frequency: "monthly, quarterly, annually"
  - payment_terms: "Net 30, Net 60, etc."
relationships:
  - belongs_to: tenant, account
  - has_many: contract_line_items, invoices
  - has_one: contract_owner (user)
```

#### **Subscription**
```yaml
entity: subscription
description: "Recurring revenue subscriptions"
attributes:
  - subscription_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_id: "Customer account"
  - contract_id: "Parent contract"
  - subscription_name: "Subscription display name"
  - status: "active, paused, cancelled, expired"
  - start_date: "Subscription start date"
  - end_date: "Subscription end date"
  - billing_cycle: "monthly, quarterly, annually"
  - mrr: "Monthly recurring revenue"
  - arr: "Annual recurring revenue"
  - quantity: "Licensed quantity"
  - unit_price: "Price per unit"
  - next_billing_date: "Next billing date"
  - cancellation_date: "If cancelled, when"
  - cancellation_reason: "Reason for cancellation"
relationships:
  - belongs_to: tenant, account, contract
  - has_many: subscription_line_items, usage_records
```

---

### **6. Billing & Financial**

#### **Invoice**
```yaml
entity: invoice
description: "Customer invoices"
attributes:
  - invoice_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_id: "Customer account"
  - contract_id: "Related contract"
  - invoice_number: "Human-readable invoice number"
  - invoice_date: "Invoice generation date"
  - due_date: "Payment due date"
  - status: "draft, sent, paid, overdue, cancelled"
  - subtotal: "Pre-tax amount"
  - tax_amount: "Tax amount"
  - total_amount: "Total amount due"
  - currency: "Currency code"
  - payment_terms: "Payment terms"
  - billing_period_start: "Billing period start"
  - billing_period_end: "Billing period end"
relationships:
  - belongs_to: tenant, account, contract
  - has_many: invoice_line_items, payments
```

#### **Payment**
```yaml
entity: payment
description: "Customer payments"
attributes:
  - payment_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_id: "Customer account"
  - invoice_id: "Related invoice"
  - payment_amount: "Payment amount"
  - currency: "Currency code"
  - payment_date: "When payment was made"
  - payment_method: "credit_card, ach, wire, check"
  - payment_status: "pending, completed, failed, refunded"
  - transaction_id: "Payment processor transaction ID"
  - reference_number: "Payment reference"
relationships:
  - belongs_to: tenant, account, invoice
  - has_many: refunds
```

#### **Refund**
```yaml
entity: refund
description: "Payment refunds"
attributes:
  - refund_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - payment_id: "Original payment"
  - refund_amount: "Refund amount"
  - currency: "Currency code"
  - refund_date: "When refund was processed"
  - refund_reason: "Reason for refund"
  - refund_status: "pending, completed, failed"
  - transaction_id: "Refund transaction ID"
relationships:
  - belongs_to: tenant, payment
```

---

### **7. Support & Service**

#### **Case**
```yaml
entity: case
description: "Customer support cases"
attributes:
  - case_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_id: "Customer account"
  - contact_id: "Primary contact"
  - case_number: "Human-readable case number"
  - subject: "Case subject line"
  - description: "Case description"
  - status: "new, in_progress, pending, resolved, closed"
  - priority: "low, medium, high, critical"
  - case_type: "question, problem, feature_request"
  - case_origin: "email, phone, web, chat"
  - created_date: "Case creation date"
  - closed_date: "Case closure date"
  - resolution: "Resolution description"
relationships:
  - belongs_to: tenant, account, contact
  - has_one: case_owner (user)
  - has_many: case_comments, sla_violations
```

#### **SLA**
```yaml
entity: sla
description: "Service Level Agreement definitions"
attributes:
  - sla_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - sla_name: "SLA display name"
  - case_priority: "Priority this SLA applies to"
  - response_time_hours: "Response time requirement"
  - resolution_time_hours: "Resolution time requirement"
  - business_hours_only: "Boolean business hours flag"
  - escalation_rules: "JSONB escalation configuration"
relationships:
  - belongs_to: tenant
  - applies_to: cases
```

---

### **8. Territory & Quota Management**

#### **Territory**
```yaml
entity: territory
description: "Sales territory definitions"
attributes:
  - territory_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - territory_name: "Territory display name"
  - territory_type: "geographic, industry, account_based"
  - parent_territory_id: "Parent territory (hierarchy)"
  - territory_rules: "JSONB territory assignment rules"
  - is_active: "Boolean active flag"
  - effective_date: "When territory becomes effective"
  - expiration_date: "When territory expires"
relationships:
  - belongs_to: tenant
  - has_many: territory_assignments, quotas
  - belongs_to: parent_territory (self-referential)
```

#### **Quota**
```yaml
entity: quota
description: "Sales quota assignments"
attributes:
  - quota_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - user_id: "Quota holder"
  - territory_id: "Territory assignment"
  - quota_period: "monthly, quarterly, annually"
  - quota_year: "Quota year"
  - quota_quarter: "Quota quarter (if applicable)"
  - quota_amount: "Quota target amount"
  - currency: "Currency code"
  - quota_type: "revenue, bookings, activities"
  - start_date: "Quota period start"
  - end_date: "Quota period end"
relationships:
  - belongs_to: tenant, user, territory
  - has_many: quota_achievements
```

#### **Forecast**
```yaml
entity: forecast
description: "Sales forecasts"
attributes:
  - forecast_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - user_id: "Forecast owner"
  - territory_id: "Territory forecast"
  - forecast_period: "monthly, quarterly, annually"
  - forecast_year: "Forecast year"
  - forecast_quarter: "Forecast quarter (if applicable)"
  - commit_amount: "Committed forecast"
  - best_case_amount: "Best case forecast"
  - pipeline_amount: "Pipeline forecast"
  - forecast_date: "When forecast was submitted"
  - is_submitted: "Boolean submission flag"
relationships:
  - belongs_to: tenant, user, territory
  - has_many: forecast_line_items
```

---

### **9. Activities & Engagement**

#### **Activity**
```yaml
entity: activity
description: "Customer engagement activities"
attributes:
  - activity_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - account_id: "Related account"
  - contact_id: "Related contact"
  - opportunity_id: "Related opportunity"
  - user_id: "Activity owner"
  - activity_type: "call, email, meeting, task, note"
  - subject: "Activity subject"
  - description: "Activity description"
  - activity_date: "When activity occurred"
  - due_date: "When activity is due"
  - status: "planned, completed, cancelled"
  - duration_minutes: "Activity duration"
  - outcome: "Activity outcome"
relationships:
  - belongs_to: tenant, account, contact, opportunity, user
```

#### **Meeting**
```yaml
entity: meeting
description: "Scheduled meetings"
attributes:
  - meeting_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - subject: "Meeting subject"
  - description: "Meeting description"
  - start_time: "Meeting start time"
  - end_time: "Meeting end time"
  - location: "Meeting location"
  - meeting_type: "demo, discovery, negotiation, check_in"
  - status: "scheduled, completed, cancelled"
  - organizer_id: "Meeting organizer user"
relationships:
  - belongs_to: tenant, organizer (user)
  - has_many: meeting_attendees
```

---

### **10. Governance & Compliance**

#### **Evidence**
```yaml
entity: evidence
description: "Governance evidence records"
attributes:
  - evidence_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - trace_id: "Related workflow trace"
  - evidence_type: "execution, compliance, audit"
  - evidence_data: "JSONB evidence content"
  - evidence_hash: "SHA-256 hash for integrity"
  - digital_signature: "Cryptographic signature"
  - created_at: "Evidence creation timestamp"
  - retention_until: "Evidence retention date"
  - compliance_framework: "SOX, GDPR, RBI, etc."
relationships:
  - belongs_to: tenant
  - links_to: workflow_trace, policy_application
```

#### **Override**
```yaml
entity: override
description: "Policy override records"
attributes:
  - override_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - user_id: "User who created override"
  - workflow_id: "Related workflow"
  - policy_id: "Policy that was overridden"
  - override_reason: "Justification for override"
  - override_type: "emergency, planned, exception"
  - approved_by: "Approver user ID"
  - approved_at: "Approval timestamp"
  - expires_at: "Override expiration"
  - risk_level: "low, medium, high, critical"
relationships:
  - belongs_to: tenant, user
  - approved_by: user
```

#### **PolicyApplied**
```yaml
entity: policy_applied
description: "Policy application records"
attributes:
  - policy_application_id: "Primary identifier (UUID)"
  - tenant_id: "Tenant isolation key"
  - policy_id: "Applied policy"
  - workflow_id: "Related workflow"
  - trace_id: "Related trace"
  - decision: "allow, deny, warn"
  - decision_reason: "Reason for decision"
  - applied_at: "When policy was applied"
  - policy_version: "Version of policy applied"
relationships:
  - belongs_to: tenant, policy, workflow
```

---

### **11. Compliance & Data Classification**

#### **ComplianceTag**
```yaml
entity: compliance_tag
description: "Compliance classification tags"
attributes:
  - tag_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - tag_name: "Tag display name"
  - tag_type: "pii, financial, confidential, public"
  - compliance_framework: "GDPR, HIPAA, SOX, RBI"
  - retention_period_days: "Data retention requirement"
  - encryption_required: "Boolean encryption flag"
  - access_restrictions: "JSONB access rules"
relationships:
  - belongs_to: tenant
  - applied_to: accounts, contacts, opportunities
```

#### **DataClass**
```yaml
entity: data_class
description: "Data classification definitions"
attributes:
  - data_class_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - class_name: "Classification name"
  - sensitivity_level: "public, internal, confidential, restricted"
  - handling_requirements: "JSONB handling rules"
  - retention_policy: "Data retention requirements"
  - encryption_requirements: "Encryption specifications"
relationships:
  - belongs_to: tenant
  - has_many: data_classifications
```

#### **ResidencyZone**
```yaml
entity: residency_zone
description: "Data residency zones"
attributes:
  - zone_id: "Primary identifier"
  - tenant_id: "Tenant isolation key"
  - zone_name: "Zone display name"
  - region_code: "US, EU, IN, APAC"
  - country_codes: "Array of country codes"
  - storage_requirements: "JSONB storage rules"
  - transfer_restrictions: "Cross-border transfer rules"
relationships:
  - belongs_to: tenant
  - governs: accounts, data_storage
```

---

## Entity Relationships

### **Core Relationship Patterns**

#### **Tenant Isolation**
```yaml
pattern: tenant_isolation
description: "Every entity belongs to a tenant"
implementation:
  - foreign_key: tenant_id
  - rls_policy: "tenant_id = current_tenant_id()"
  - index: "btree(tenant_id)"
```

#### **Audit Trail**
```yaml
pattern: audit_trail
description: "Track creation and modification"
implementation:
  - created_at: "TIMESTAMPTZ DEFAULT NOW()"
  - created_by: "INTEGER REFERENCES users(user_id)"
  - updated_at: "TIMESTAMPTZ DEFAULT NOW()"
  - updated_by: "INTEGER REFERENCES users(user_id)"
  - deleted_at: "TIMESTAMPTZ (soft delete)"
```

#### **External System Integration**
```yaml
pattern: external_integration
description: "Link to external systems"
implementation:
  - external_id: "VARCHAR(255)"
  - source_system: "VARCHAR(50)"
  - last_sync_at: "TIMESTAMPTZ"
  - sync_status: "success, failed, pending"
```

#### **Hierarchical Relationships**
```yaml
pattern: hierarchy
description: "Parent-child relationships"
implementation:
  - parent_id: "UUID REFERENCES same_table(id)"
  - hierarchy_path: "LTREE for efficient queries"
  - level: "INTEGER for depth tracking"
```

---

## Industry-Specific Extensions

### **SaaS Industry**
```yaml
extensions:
  account:
    - mrr: "Monthly recurring revenue"
    - arr: "Annual recurring revenue"
    - plan_type: "free, starter, professional, enterprise"
    - usage_metrics: "JSONB usage data"
  
  subscription:
    - feature_flags: "JSONB enabled features"
    - usage_limits: "JSONB usage quotas"
    - overage_charges: "DECIMAL overage amounts"
```

### **Banking Industry**
```yaml
extensions:
  account:
    - account_number: "Bank account number"
    - account_type: "savings, current, loan, credit"
    - kyc_status: "completed, pending, expired"
    - risk_rating: "low, medium, high, very_high"
  
  transaction:
    - transaction_type: "debit, credit, transfer"
    - transaction_amount: "DECIMAL amount"
    - currency_code: "ISO currency code"
    - regulatory_flags: "JSONB compliance flags"
```

### **Insurance Industry**
```yaml
extensions:
  account:
    - policy_number: "Insurance policy number"
    - policy_type: "life, health, auto, property"
    - premium_amount: "DECIMAL premium"
    - coverage_amount: "DECIMAL coverage"
  
  claim:
    - claim_number: "Claim identifier"
    - claim_amount: "DECIMAL claim amount"
    - claim_status: "submitted, processing, approved, denied"
    - adjuster_id: "Assigned adjuster"
```

---

## Data Quality Rules

### **Mandatory Fields**
```yaml
rules:
  all_entities:
    - tenant_id: "NOT NULL"
    - created_at: "NOT NULL DEFAULT NOW()"
    - updated_at: "NOT NULL DEFAULT NOW()"
  
  customer_entities:
    - external_id: "NOT NULL (for CRM sync)"
    - source_system: "NOT NULL"
  
  financial_entities:
    - currency: "NOT NULL DEFAULT 'USD'"
    - amount: "CHECK (amount >= 0)"
```

### **Data Validation**
```yaml
validation:
  email_format:
    - pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
  
  phone_format:
    - pattern: "^\\+?[1-9]\\d{1,14}$"
  
  currency_codes:
    - enum: ["USD", "EUR", "GBP", "INR", "CAD", "AUD"]
  
  status_values:
    - account_status: ["active", "inactive", "suspended"]
    - opportunity_status: ["open", "won", "lost", "cancelled"]
```

### **Referential Integrity**
```yaml
constraints:
  cascade_deletes:
    - account -> contacts: "CASCADE"
    - account -> opportunities: "RESTRICT"
    - opportunity -> activities: "CASCADE"
  
  required_references:
    - opportunity.account_id: "NOT NULL"
    - contact.account_id: "NOT NULL"
    - activity.user_id: "NOT NULL"
```

---

## Performance Considerations

### **Indexing Strategy**
```yaml
indexes:
  tenant_isolation:
    - "btree(tenant_id)" on all tables
  
  common_queries:
    - "btree(account_id, created_at DESC)" for activities
    - "btree(user_id, activity_date DESC)" for user activities
    - "btree(close_date, stage)" for opportunities
  
  full_text_search:
    - "gin(to_tsvector('english', account_name))" for accounts
    - "gin(to_tsvector('english', subject || ' ' || description))" for cases
```

### **Partitioning Strategy**
```yaml
partitioning:
  time_series_data:
    - activities: "PARTITION BY RANGE (activity_date)"
    - invoices: "PARTITION BY RANGE (invoice_date)"
    - payments: "PARTITION BY RANGE (payment_date)"
  
  tenant_partitioning:
    - large_tenants: "PARTITION BY LIST (tenant_id)"
    - retention: "monthly partitions for easy archival"
```

---

## Governance & Compliance

### **Data Lineage**
```yaml
lineage:
  source_systems:
    - salesforce: "accounts, contacts, opportunities"
    - stripe: "invoices, payments, subscriptions"
    - zendesk: "cases, sla_violations"
  
  transformation_rules:
    - standardization: "phone numbers, addresses"
    - enrichment: "industry classification, company size"
    - validation: "email format, required fields"
```

### **Privacy & Security**
```yaml
privacy:
  pii_fields:
    - contact.email: "encrypted, masked in non-prod"
    - contact.phone: "encrypted, masked in non-prod"
    - account.billing_address: "encrypted"
  
  retention_policies:
    - contact_data: "7 years after last activity"
    - financial_data: "10 years (SOX requirement)"
    - support_data: "5 years after case closure"
```

---

## Usage Guidelines

### **For Developers**
1. **Always include tenant_id** in queries and constraints
2. **Use UUIDs for external-facing IDs** to prevent enumeration
3. **Include audit columns** in all business entities
4. **Validate foreign key relationships** before insertion
5. **Use appropriate indexes** for query patterns

### **For Data Analysts**
1. **Join through account_id** for customer-centric analysis
2. **Filter by tenant_id** for multi-tenant isolation
3. **Use created_at/updated_at** for temporal analysis
4. **Respect data classification** and privacy rules
5. **Document data lineage** for regulatory compliance

### **For Product Managers**
1. **Understand entity relationships** for feature planning
2. **Consider industry extensions** for market expansion
3. **Plan for data retention** and compliance requirements
4. **Design for multi-tenancy** from the start
5. **Include governance hooks** in new features

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** data-architecture@company.com
