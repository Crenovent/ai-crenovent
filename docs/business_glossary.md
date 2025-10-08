# RevOps Business Glossary & Data Dictionary

**Task 7.2-T52: Create dictionary/glossary for metrics & fields**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** RevOps Platform Team

---

## Overview

This business glossary provides standardized definitions for all metrics, fields, and business terms used across the RevOps platform. It ensures consistent understanding and usage across teams, systems, and stakeholders.

---

## Core Business Entities

### Account
**Definition:** A company or organization that represents a potential or existing customer.  
**Database Table:** `accounts`  
**Key Fields:**
- `account_id` (UUID): Unique identifier for the account
- `account_name` (VARCHAR): Official company name
- `account_type` (ENUM): Classification - prospect, customer, partner, competitor
- `annual_revenue` (DECIMAL): Company's annual revenue in USD
- `employee_count` (INTEGER): Number of employees

**Business Rules:**
- Account names must be unique within a tenant
- Annual revenue must be >= 0
- Account type determines available workflows

### Opportunity
**Definition:** A potential sales deal with a specific account that has a probability of closing.  
**Database Table:** `opportunities`  
**Key Fields:**
- `opportunity_id` (UUID): Unique identifier for the opportunity
- `opportunity_name` (VARCHAR): Descriptive name for the deal
- `amount` (DECIMAL): Expected deal value in USD
- `probability` (DECIMAL): Likelihood of closing (0-100%)
- `stage` (VARCHAR): Current position in sales process
- `close_date` (DATE): Expected or actual close date

**Business Rules:**
- Probability must be between 0 and 100
- Amount must be >= 0
- Stage must follow defined sales process
- Close date cannot be in the past for open opportunities

### Tenant
**Definition:** An isolated instance of the platform for a specific organization.  
**Database Table:** `tenant_metadata`  
**Key Fields:**
- `tenant_id` (INTEGER): Unique identifier for the tenant
- `tenant_name` (VARCHAR): Organization name
- `industry_code` (ENUM): Industry classification
- `region_code` (ENUM): Geographic region
- `compliance_requirements` (JSONB): Required compliance frameworks

**Business Rules:**
- Each tenant has complete data isolation
- Industry code determines available templates
- Region code affects data residency requirements

---

## Revenue Metrics

### Annual Recurring Revenue (ARR)
**Definition:** The yearly value of recurring subscription revenue.  
**Calculation:** `SUM(monthly_recurring_revenue * 12) WHERE subscription_status = 'active'`  
**Unit:** USD  
**Frequency:** Monthly calculation  
**Business Context:** Primary growth metric for SaaS companies

### Monthly Recurring Revenue (MRR)
**Definition:** The monthly value of recurring subscription revenue.  
**Calculation:** `SUM(subscription_amount) WHERE subscription_status = 'active' AND billing_frequency = 'monthly'`  
**Unit:** USD  
**Frequency:** Daily calculation  
**Business Context:** Month-over-month growth tracking

### Customer Acquisition Cost (CAC)
**Definition:** The total cost to acquire a new customer.  
**Calculation:** `total_sales_marketing_spend / new_customers_acquired`  
**Unit:** USD  
**Frequency:** Monthly calculation  
**Business Context:** Efficiency of customer acquisition efforts

### Customer Lifetime Value (CLV)
**Definition:** The predicted total revenue from a customer relationship.  
**Calculation:** `(average_revenue_per_customer * gross_margin) / churn_rate`  
**Unit:** USD  
**Frequency:** Quarterly calculation  
**Business Context:** Long-term customer value assessment

### Churn Rate
**Definition:** The percentage of customers who cancel their subscription in a given period.  
**Calculation:** `(customers_lost / customers_at_start_of_period) * 100`  
**Unit:** Percentage  
**Frequency:** Monthly calculation  
**Business Context:** Customer retention health indicator

---

## Sales Metrics

### Pipeline Coverage
**Definition:** The ratio of pipeline value to quota or target.  
**Calculation:** `total_pipeline_value / quota_amount`  
**Unit:** Ratio (e.g., 3.2x)  
**Frequency:** Weekly calculation  
**Business Context:** Predictive indicator of quota attainment

### Win Rate
**Definition:** The percentage of opportunities that result in closed-won deals.  
**Calculation:** `(closed_won_opportunities / total_closed_opportunities) * 100`  
**Unit:** Percentage  
**Frequency:** Monthly calculation  
**Business Context:** Sales effectiveness measurement

### Average Deal Size
**Definition:** The mean value of closed-won opportunities.  
**Calculation:** `SUM(closed_won_amount) / COUNT(closed_won_opportunities)`  
**Unit:** USD  
**Frequency:** Monthly calculation  
**Business Context:** Deal size trends and forecasting

### Sales Cycle Length
**Definition:** The average time from opportunity creation to close.  
**Calculation:** `AVG(close_date - created_date) WHERE stage = 'closed_won'`  
**Unit:** Days  
**Frequency:** Monthly calculation  
**Business Context:** Sales process efficiency

---

## Operational Metrics

### Data Quality Score
**Definition:** The percentage of records that meet defined quality standards.  
**Calculation:** `(records_passing_quality_checks / total_records) * 100`  
**Unit:** Percentage  
**Frequency:** Daily calculation  
**Business Context:** Data reliability for automation

### Workflow Success Rate
**Definition:** The percentage of workflow executions that complete successfully.  
**Calculation:** `(successful_executions / total_executions) * 100`  
**Unit:** Percentage  
**Frequency:** Real-time calculation  
**Business Context:** Automation reliability

### Policy Compliance Rate
**Definition:** The percentage of workflows that execute without policy violations.  
**Calculation:** `(compliant_executions / total_executions) * 100`  
**Unit:** Percentage  
**Frequency:** Real-time calculation  
**Business Context:** Governance effectiveness

---

## Governance & Compliance Terms

### Evidence Pack
**Definition:** An immutable record of workflow execution including inputs, outputs, and governance metadata.  
**Purpose:** Audit trail and compliance documentation  
**Retention:** Varies by industry (90 days to 7 years)  
**Access:** Restricted to authorized personnel and auditors

### Override Ledger
**Definition:** A tamper-proof log of all manual interventions in automated processes.  
**Purpose:** Accountability and audit trail for exceptions  
**Required Fields:** Who, what, when, why, approval chain  
**Access:** Audit-only access with segregation of duties

### Policy Pack
**Definition:** A collection of governance rules applied to workflows.  
**Scope:** Tenant, industry, or regulatory framework specific  
**Examples:** SOX financial controls, GDPR data protection, RBI banking regulations  
**Enforcement:** Automatic at workflow execution time

### Trust Score
**Definition:** A calculated measure of workflow reliability and compliance.  
**Range:** 0.0 to 1.0 (higher is better)  
**Factors:** Success rate, policy compliance, override frequency, data quality  
**Usage:** Workflow promotion and risk assessment

---

## Industry-Specific Terms

### SaaS Industry

#### Subscription
**Definition:** A recurring revenue agreement with defined terms and billing cycle.  
**Key Attributes:** Start date, end date, billing frequency, amount, status  
**States:** Active, paused, cancelled, expired  
**Business Impact:** Direct ARR/MRR contribution

#### Expansion Revenue
**Definition:** Additional revenue from existing customers through upsells or cross-sells.  
**Calculation:** `current_period_revenue - previous_period_revenue WHERE customer_exists`  
**Tracking:** Monthly cohort analysis  
**Business Context:** Growth efficiency and customer success

### Banking Industry

#### Loan Sanction
**Definition:** Formal approval of a loan application with specified terms.  
**Required Approvals:** Credit committee, risk assessment, compliance review  
**Documentation:** Evidence pack with all approval signatures  
**Regulatory:** Must comply with RBI guidelines

#### Disbursal
**Definition:** The actual transfer of approved loan funds to the borrower.  
**Prerequisites:** Completed sanction, documentation, compliance checks  
**Tracking:** Amount, date, method, beneficiary details  
**Audit Trail:** Full evidence pack required

### Insurance Industry

#### Claim
**Definition:** A request for compensation under an insurance policy.  
**Lifecycle:** Reported → Investigated → Assessed → Settled/Denied  
**Key Metrics:** Settlement ratio, average settlement time, claim amount  
**Compliance:** IRDAI regulations and audit requirements

#### Underwriting Decision
**Definition:** The assessment and pricing of insurance risk.  
**Factors:** Risk profile, coverage amount, policy terms, regulatory requirements  
**Documentation:** Risk assessment, pricing model, approval chain  
**Governance:** Must follow established underwriting guidelines

---

## Data Classification

### Public Data
**Definition:** Information that can be freely shared without restriction.  
**Examples:** Company name, public website, industry classification  
**Protection Level:** None required  
**Access:** Unrestricted

### Internal Data
**Definition:** Information intended for internal use within the organization.  
**Examples:** Sales targets, internal processes, employee directories  
**Protection Level:** Access controls and authentication  
**Access:** Employees and authorized contractors

### Confidential Data
**Definition:** Sensitive information that could harm the organization if disclosed.  
**Examples:** Customer data, financial records, strategic plans  
**Protection Level:** Encryption, access logging, need-to-know basis  
**Access:** Authorized personnel only

### Restricted Data
**Definition:** Highly sensitive information with legal or regulatory protection requirements.  
**Examples:** PII, PHI, financial account numbers, social security numbers  
**Protection Level:** Encryption, masking, audit trails, regulatory compliance  
**Access:** Minimal access with approval workflows

---

## Compliance Frameworks

### SOX (Sarbanes-Oxley Act)
**Scope:** Financial reporting and internal controls  
**Requirements:** Segregation of duties, audit trails, management certification  
**Platform Impact:** Evidence packs, override ledger, approval workflows  
**Retention:** 7 years minimum

### GDPR (General Data Protection Regulation)
**Scope:** Personal data protection for EU residents  
**Requirements:** Consent management, data minimization, right to erasure  
**Platform Impact:** Data classification, masking, retention policies  
**Penalties:** Up to 4% of annual revenue

### HIPAA (Health Insurance Portability and Accountability Act)
**Scope:** Protected health information (PHI)  
**Requirements:** Access controls, audit logs, breach notification  
**Platform Impact:** Encryption, access logging, data masking  
**Compliance:** Business associate agreements required

### RBI (Reserve Bank of India)
**Scope:** Banking operations and data localization  
**Requirements:** Data residency, audit trails, risk management  
**Platform Impact:** Geographic data storage, compliance overlays  
**Reporting:** Regular regulatory submissions required

---

## Technical Terms

### Row Level Security (RLS)
**Definition:** Database security feature that controls access to rows based on user characteristics.  
**Implementation:** PostgreSQL policies using tenant_id filtering  
**Purpose:** Multi-tenant data isolation  
**Performance:** Minimal overhead with proper indexing

### UUID (Universally Unique Identifier)
**Definition:** 128-bit identifier that is unique across time and space.  
**Format:** Version 4 (random) or Version 7 (time-ordered)  
**Usage:** Primary keys for all major entities  
**Benefits:** No collision risk, globally unique, sortable (v7)

### JSONB
**Definition:** PostgreSQL binary JSON data type with indexing support.  
**Usage:** Flexible schema fields, metadata, configuration  
**Indexing:** GIN indexes for efficient queries  
**Benefits:** Schema flexibility with performance

---

## Calculation Standards

### Currency Handling
- All monetary values stored in USD with 2 decimal precision
- Exchange rates applied at transaction time
- Historical rates preserved for audit purposes
- Rounding follows banker's rounding (round half to even)

### Date/Time Standards
- All timestamps in UTC (TIMESTAMPTZ)
- Business dates in local timezone of tenant
- Fiscal year alignment per tenant configuration
- Date ranges inclusive of start, exclusive of end

### Percentage Calculations
- Stored as decimal values (0.0 to 1.0)
- Displayed as percentages (0% to 100%)
- Precision to 2 decimal places in display
- Null values treated as 0% unless specified

---

## Approval and Maintenance

**Approved By:** RevOps Platform Team  
**Review Cycle:** Quarterly  
**Change Process:** Pull request with stakeholder review  
**Distribution:** Available to all platform users  
**Questions:** Contact platform-team@company.com

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025
