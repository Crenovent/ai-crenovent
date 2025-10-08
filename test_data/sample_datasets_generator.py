# ai-crenovent/test_data/sample_datasets_generator.py
"""
Task 7.2-T47: Provide sample datasets per industry - CSV/Parquet
Generate industry-specific sample datasets for development and testing
"""

import csv
import json
import os
import random
import uuid
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Industry(Enum):
    SAAS = "SaaS"
    BANKING = "BANK"
    INSURANCE = "INSUR"
    ECOMMERCE = "ECOMM"
    FINTECH = "FS"
    IT_SERVICES = "IT"

class Region(Enum):
    US = "US"
    EU = "EU"
    INDIA = "IN"
    APAC = "APAC"

@dataclass
class SampleTenant:
    tenant_id: int
    tenant_name: str
    industry_code: str
    region_code: str
    compliance_requirements: List[str]
    status: str = "active"
    created_at: str = ""
    metadata: Dict[str, Any] = None

@dataclass
class SampleAccount:
    account_id: str
    tenant_id: int
    external_id: str
    account_name: str
    account_type: str
    industry: str
    annual_revenue: float
    employee_count: int
    website: str
    created_at: str
    is_deleted: bool = False

@dataclass
class SampleOpportunity:
    opportunity_id: str
    tenant_id: int
    account_id: str
    external_id: str
    opportunity_name: str
    stage: str
    amount: float
    probability: float
    close_date: str
    owner_user_id: int
    lead_source: str
    created_at: str
    is_deleted: bool = False

class SampleDatasetGenerator:
    """
    Task 7.2-T47: Generate sample datasets per industry
    Creates realistic test data for different industry verticals
    """
    
    def __init__(self, output_dir: str = "ai-crenovent/test_data/samples"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Industry-specific configurations
        self.industry_configs = {
            Industry.SAAS: {
                "account_types": ["prospect", "customer", "partner"],
                "industries": ["Software", "Technology", "SaaS", "Cloud Services"],
                "stages": ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"],
                "lead_sources": ["Website", "Referral", "Content Marketing", "Demo Request", "Trial Signup"],
                "compliance": ["SOX", "GDPR"],
                "revenue_range": (10000, 50000000),
                "employee_range": (10, 5000)
            },
            Industry.BANKING: {
                "account_types": ["prospect", "customer"],
                "industries": ["Banking", "Financial Services", "Credit Union", "Investment"],
                "stages": ["Application", "Documentation", "Credit Review", "Approval", "Sanctioned", "Disbursed", "Rejected"],
                "lead_sources": ["Branch Visit", "Online Application", "Referral", "Agent"],
                "compliance": ["SOX", "RBI", "BASEL"],
                "revenue_range": (1000000, 100000000),
                "employee_range": (100, 10000)
            },
            Industry.INSURANCE: {
                "account_types": ["prospect", "customer", "broker"],
                "industries": ["Insurance", "Reinsurance", "Brokerage", "Risk Management"],
                "stages": ["Inquiry", "Quote", "Underwriting", "Policy Issued", "Claim Filed", "Claim Settled"],
                "lead_sources": ["Agent", "Broker", "Online", "Referral"],
                "compliance": ["IRDAI", "GDPR", "SOX"],
                "revenue_range": (500000, 20000000),
                "employee_range": (50, 2000)
            }
        }
        
        # Sample company names by industry
        self.company_names = {
            Industry.SAAS: [
                "CloudTech Solutions", "DataFlow Systems", "SaaS Innovations", "Digital Workspace",
                "Analytics Pro", "Workflow Automation", "Customer Success Platform", "Revenue Intelligence",
                "Sales Enablement Hub", "Marketing Automation Co", "HR Tech Solutions", "FinTech Analytics"
            ],
            Industry.BANKING: [
                "First National Bank", "Community Credit Union", "Regional Banking Corp", "Metro Financial",
                "Capital Trust Bank", "Savings & Loan Association", "Investment Banking Group", "Credit Solutions",
                "Mortgage Lending Corp", "Commercial Banking Ltd", "Private Wealth Management", "Digital Bank"
            ],
            Industry.INSURANCE: [
                "Reliable Insurance Co", "Protection Plus", "Risk Management Solutions", "Family Insurance Group",
                "Commercial Coverage Corp", "Life & Health Insurance", "Property Casualty Inc", "Reinsurance Partners",
                "Claims Management Co", "Underwriting Specialists", "Broker Network Ltd", "Insurance Technology"
            ]
        }
    
    def generate_tenants(self, count: int = 10) -> List[SampleTenant]:
        """Generate sample tenant data"""
        tenants = []
        
        for i in range(count):
            industry = random.choice(list(Industry))
            region = random.choice(list(Region))
            config = self.industry_configs.get(industry, self.industry_configs[Industry.SAAS])
            
            tenant = SampleTenant(
                tenant_id=1000 + i,
                tenant_name=f"{industry.value} Demo Company {i+1}",
                industry_code=industry.value,
                region_code=region.value,
                compliance_requirements=config["compliance"],
                created_at=self._random_datetime_str(days_back=365),
                metadata={"demo": True, "created_by": "sample_generator"}
            )
            tenants.append(tenant)
        
        return tenants
    
    def generate_accounts(self, tenants: List[SampleTenant], accounts_per_tenant: int = 20) -> List[SampleAccount]:
        """Generate sample account data"""
        accounts = []
        
        for tenant in tenants:
            industry_enum = Industry(tenant.industry_code)
            config = self.industry_configs.get(industry_enum, self.industry_configs[Industry.SAAS])
            company_names = self.company_names.get(industry_enum, self.company_names[Industry.SAAS])
            
            for i in range(accounts_per_tenant):
                account = SampleAccount(
                    account_id=str(uuid.uuid4()),
                    tenant_id=tenant.tenant_id,
                    external_id=f"EXT_{tenant.tenant_id}_{i+1:04d}",
                    account_name=f"{random.choice(company_names)} {random.randint(1, 999)}",
                    account_type=random.choice(config["account_types"]),
                    industry=random.choice(config["industries"]),
                    annual_revenue=random.uniform(*config["revenue_range"]),
                    employee_count=random.randint(*config["employee_range"]),
                    website=f"https://www.{self._slugify(company_names[i % len(company_names)])}.com",
                    created_at=self._random_datetime_str(days_back=730)
                )
                accounts.append(account)
        
        return accounts
    
    def generate_opportunities(self, accounts: List[SampleAccount], opps_per_account: int = 3) -> List[SampleOpportunity]:
        """Generate sample opportunity data"""
        opportunities = []
        
        for account in accounts:
            # Get industry config
            tenant_industry = None
            for industry in Industry:
                if industry.value in account.industry or industry.value == account.account_name[:4]:
                    tenant_industry = industry
                    break
            
            if not tenant_industry:
                tenant_industry = Industry.SAAS
            
            config = self.industry_configs.get(tenant_industry, self.industry_configs[Industry.SAAS])
            
            for i in range(random.randint(1, opps_per_account)):
                # Generate opportunity amount based on account size
                base_amount = account.annual_revenue * 0.1  # 10% of annual revenue
                amount = random.uniform(base_amount * 0.1, base_amount * 0.5)
                
                # Assign probability based on stage
                stage = random.choice(config["stages"])
                probability = self._get_probability_for_stage(stage, tenant_industry)
                
                # Generate close date
                close_date = self._random_future_date(days_ahead=180)
                
                opportunity = SampleOpportunity(
                    opportunity_id=str(uuid.uuid4()),
                    tenant_id=account.tenant_id,
                    account_id=account.account_id,
                    external_id=f"OPP_{account.tenant_id}_{i+1:06d}",
                    opportunity_name=f"{account.account_name} - {self._get_opportunity_suffix(tenant_industry)}",
                    stage=stage,
                    amount=round(amount, 2),
                    probability=probability,
                    close_date=close_date.isoformat(),
                    owner_user_id=random.randint(1300, 1330),  # Sample user IDs
                    lead_source=random.choice(config["lead_sources"]),
                    created_at=self._random_datetime_str(days_back=90)
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug"""
        return text.lower().replace(" ", "").replace("&", "and")
    
    def _random_datetime_str(self, days_back: int = 30) -> str:
        """Generate random datetime string in the past"""
        days_ago = random.randint(1, days_back)
        dt = datetime.now() - timedelta(days=days_ago)
        return dt.isoformat()
    
    def _random_future_date(self, days_ahead: int = 90) -> date:
        """Generate random future date"""
        days_forward = random.randint(1, days_ahead)
        return date.today() + timedelta(days=days_forward)
    
    def _get_probability_for_stage(self, stage: str, industry: Industry) -> float:
        """Get realistic probability based on stage and industry"""
        stage_probabilities = {
            # SaaS stages
            "Prospecting": 10.0,
            "Qualification": 25.0,
            "Proposal": 50.0,
            "Negotiation": 75.0,
            "Closed Won": 100.0,
            "Closed Lost": 0.0,
            
            # Banking stages
            "Application": 20.0,
            "Documentation": 40.0,
            "Credit Review": 60.0,
            "Approval": 80.0,
            "Sanctioned": 95.0,
            "Disbursed": 100.0,
            "Rejected": 0.0,
            
            # Insurance stages
            "Inquiry": 15.0,
            "Quote": 35.0,
            "Underwriting": 65.0,
            "Policy Issued": 100.0,
            "Claim Filed": 90.0,
            "Claim Settled": 100.0
        }
        
        return stage_probabilities.get(stage, 50.0)
    
    def _get_opportunity_suffix(self, industry: Industry) -> str:
        """Get industry-specific opportunity name suffix"""
        suffixes = {
            Industry.SAAS: ["Annual Subscription", "Platform License", "Enterprise Plan", "Professional Services"],
            Industry.BANKING: ["Business Loan", "Credit Line", "Mortgage", "Investment Account"],
            Industry.INSURANCE: ["Commercial Policy", "Life Insurance", "Property Coverage", "Liability Plan"]
        }
        
        return random.choice(suffixes.get(industry, suffixes[Industry.SAAS]))
    
    def save_to_csv(self, data: List[Any], filename: str, fieldnames: List[str]) -> str:
        """Save data to CSV file"""
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in data:
                if hasattr(item, '__dict__'):
                    row = asdict(item)
                else:
                    row = item
                
                # Handle complex fields
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value)
                
                writer.writerow(row)
        
        logger.info(f"Saved {len(data)} records to {file_path}")
        return file_path
    
    def generate_industry_dataset(self, industry: Industry, tenant_count: int = 3) -> Dict[str, str]:
        """Generate complete dataset for a specific industry"""
        # Filter tenants for this industry
        all_tenants = self.generate_tenants(count=10)
        industry_tenants = [t for t in all_tenants if t.industry_code == industry.value][:tenant_count]
        
        if not industry_tenants:
            # Create at least one tenant for this industry
            industry_tenants = [SampleTenant(
                tenant_id=2000 + industry.value.__hash__() % 1000,
                tenant_name=f"{industry.value} Sample Company",
                industry_code=industry.value,
                region_code=Region.US.value,
                compliance_requirements=self.industry_configs[industry]["compliance"],
                created_at=self._random_datetime_str(days_back=365),
                metadata={"demo": True, "industry_specific": True}
            )]
        
        # Generate accounts and opportunities
        accounts = self.generate_accounts(industry_tenants, accounts_per_tenant=15)
        opportunities = self.generate_opportunities(accounts, opps_per_account=4)
        
        # Save to CSV files
        files_created = {}
        
        # Tenants
        tenant_file = f"{industry.value.lower()}_tenants.csv"
        files_created['tenants'] = self.save_to_csv(
            industry_tenants, 
            tenant_file,
            ['tenant_id', 'tenant_name', 'industry_code', 'region_code', 'compliance_requirements', 'status', 'created_at', 'metadata']
        )
        
        # Accounts
        account_file = f"{industry.value.lower()}_accounts.csv"
        files_created['accounts'] = self.save_to_csv(
            accounts,
            account_file,
            ['account_id', 'tenant_id', 'external_id', 'account_name', 'account_type', 'industry', 'annual_revenue', 'employee_count', 'website', 'created_at', 'is_deleted']
        )
        
        # Opportunities
        opp_file = f"{industry.value.lower()}_opportunities.csv"
        files_created['opportunities'] = self.save_to_csv(
            opportunities,
            opp_file,
            ['opportunity_id', 'tenant_id', 'account_id', 'external_id', 'opportunity_name', 'stage', 'amount', 'probability', 'close_date', 'owner_user_id', 'lead_source', 'created_at', 'is_deleted']
        )
        
        return files_created
    
    def generate_all_industry_datasets(self) -> Dict[str, Dict[str, str]]:
        """Generate datasets for all supported industries"""
        all_files = {}
        
        for industry in [Industry.SAAS, Industry.BANKING, Industry.INSURANCE]:
            print(f"Generating {industry.value} dataset...")
            files = self.generate_industry_dataset(industry)
            all_files[industry.value] = files
            print(f"✅ {industry.value} dataset complete: {len(files)} files")
        
        return all_files
    
    def create_readme(self) -> str:
        """Create README file for sample datasets"""
        readme_content = """# Sample Datasets for RevOps Platform

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

**Generated:** {generation_date}  
**Version:** 1.0.0  
**Contact:** platform-team@company.com
"""
        
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content.format(generation_date=datetime.now().isoformat()))
        
        return readme_path

# Example usage
if __name__ == "__main__":
    generator = SampleDatasetGenerator()
    
    # Generate all industry datasets
    all_files = generator.generate_all_industry_datasets()
    
    # Create README
    readme_path = generator.create_readme()
    
    print(f"\n✅ Task 7.2-T47 completed: Sample datasets generated")
    print(f"📁 Output directory: {generator.output_dir}")
    print(f"📄 README: {readme_path}")
    print(f"🏭 Industries: {', '.join(all_files.keys())}")
    
    total_files = sum(len(files) for files in all_files.values())
    print(f"📊 Total files: {total_files + 1}")  # +1 for README
