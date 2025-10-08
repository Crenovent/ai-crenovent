# Sample Datasets for RevOps Platform

**Task 7.2-T47: Provide sample datasets per industry**

This directory contains industry-specific sample datasets for development and testing.

## Available Industries

### SaaS Industry
- **Files:** saas_tenants.csv, saas_accounts.csv, saas_opportunities.csv
- **Focus:** Subscription-based software companies
- **Stages:** Prospecting → Qualification → Proposal → Negotiation → Closed Won/Lost
- **Compliance:** SOX, GDPR
- **Metrics:** ARR, MRR, Churn Rate, CAC, CLV

### Banking Industry
- **Files:** bank_tenants.csv, bank_accounts.csv, bank_opportunities.csv
- **Focus:** Financial institutions and lending
- **Stages:** Application → Documentation → Credit Review → Approval → Sanctioned → Disbursed
- **Compliance:** SOX, RBI, BASEL
- **Metrics:** Loan portfolio, Default rates, Interest margins

### Insurance Industry
- **Files:** insur_tenants.csv, insur_accounts.csv, insur_opportunities.csv
- **Focus:** Insurance companies and brokers
- **Stages:** Inquiry → Quote → Underwriting → Policy Issued → Claims
- **Compliance:** IRDAI, GDPR, SOX
- **Metrics:** Premium volume, Claims ratio, Underwriting profit

## Data Characteristics

- **Realistic Values:** Revenue, employee counts, and deal sizes based on industry norms
- **Multi-Tenant:** Each dataset includes multiple tenants with proper isolation
- **Temporal Data:** Created dates, close dates, and lifecycle progression
- **Compliance Ready:** Industry-specific compliance requirements included
- **Referential Integrity:** Proper foreign key relationships maintained

## Usage

1. **Development Testing:** Load datasets into local development environment
2. **Demo Scenarios:** Use for customer demonstrations and proof-of-concepts
3. **Performance Testing:** Scale datasets for load testing scenarios
4. **Training Data:** Use for ML model training and validation

## File Formats

- **CSV Format:** Standard comma-separated values with headers
- **UTF-8 Encoding:** Supports international characters
- **JSON Fields:** Complex fields (metadata, compliance) stored as JSON strings

## Loading Instructions

```sql
-- Example: Load SaaS tenants
COPY tenant_metadata FROM 'saas_tenants.csv' DELIMITER ',' CSV HEADER;

-- Example: Load SaaS accounts
COPY accounts FROM 'saas_accounts.csv' DELIMITER ',' CSV HEADER;

-- Example: Load SaaS opportunities
COPY opportunities FROM 'saas_opportunities.csv' DELIMITER ',' CSV HEADER;
```

## Data Privacy

- **No Real Data:** All data is synthetically generated
- **No PII:** No personally identifiable information included
- **Safe for Testing:** Can be used in any environment without privacy concerns

## Regeneration

To regenerate datasets with new random data:

```bash
cd ai-crenovent
python test_data/sample_datasets_generator.py
```

---

**Generated:** 2025-10-08T14:25:03.960528  
**Version:** 1.0.0  
**Contact:** platform-team@company.com
