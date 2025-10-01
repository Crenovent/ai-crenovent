# Test CSV Hierarchy Structures - Real-World Organizations

## Overview
This collection contains 8 different CSV files representing diverse organizational structures found in real-world companies. Each CSV has varying user counts, hierarchical depths, and structural patterns to thoroughly test the hierarchy processing system.

---

## 1. **Startup Company - TechFlow Innovations** 
**File:** `startup_techflow_flat.csv`
- **Users:** 9 employees
- **Structure:** Flat hierarchy (2-3 levels max)
- **Org Leader:** Sarah Kim (CEO & Founder)

### Hierarchy Structure:
```
Sarah Kim (CEO) [ORG LEADER]
├── Marcus Chen (CTO)
│   ├── James Wilson (Senior Developer)
│   └── Emma Thompson (Full Stack Developer)
├── Lisa Rodriguez (Head of Product)
│   ├── David Park (Product Manager)
│   └── Rachel Green (UX Designer)
├── Alex Johnson (Sales Lead)
└── Maya Patel (Marketing Manager)
```

**Characteristics:**
- Typical startup with founder as CEO
- Direct reporting to leadership
- Cross-functional teams
- Remote-friendly structure

---

## 2. **Small-Medium Business - GrowthCorp Solutions**
**File:** `smb_growthcorp_3level.csv`
- **Users:** 30 employees
- **Structure:** 3-level hierarchy (CEO → VP → Manager → Individual Contributor)
- **Org Leader:** Michael Thompson (CEO)

### Hierarchy Structure:
```
Michael Thompson (CEO) [ORG LEADER]
├── Jennifer Walsh (VP of Sales)
│   ├── Thomas Anderson (Sales Director - East)
│   │   ├── Kevin Zhang (Senior Sales Rep)
│   │   └── Lisa Garcia (Senior Sales Rep)
│   ├── Sarah Johnson (Sales Director - West)
│   │   ├── James Rodriguez (Sales Rep)
│   │   └── Amy Liu (Sales Rep)
│   └── Olivia Martinez (Customer Success Manager)
├── Robert Kim (VP of Engineering)
│   ├── Mark Wilson (Engineering Manager)
│   │   ├── Chris Taylor (Senior Developer)
│   │   ├── Emily White (Software Developer)
│   │   ├── Ryan Murphy (Frontend Developer)
│   │   └── Alex Johnson (QA Engineer)
│   └── Jessica Brown (Product Manager)
│       └── Sophie Turner (UX Designer)
├── Amanda Davis (VP of Marketing)
│   ├── Daniel Lee (Marketing Manager - Digital)
│   │   ├── Maria Santos (Digital Marketing Specialist)
│   │   └── Brian Cooper (SEO Specialist)
│   └── Rachel Miller (Marketing Manager - Content)
│       ├── Ashley Parker (Content Writer)
│       └── Nathan Brooks (Social Media Manager)
├── Carlos Martinez (CFO)
│   ├── Victoria Adams (Financial Analyst)
│   └── Jordan Hill (Accountant)
└── Linda Chen (Head of HR)
    ├── Grace Wang (HR Specialist)
    └── Tyler Scott (Recruiter)
```

**Characteristics:**
- Clear departmental structure
- Geographic sales division (East/West)
- Specialized roles within departments
- Mix of senior and mid-level positions

---

## 3. **Enterprise Corporation - GlobalTech Enterprises**
**File:** `enterprise_globaltech_5level.csv`
- **Users:** 50 employees
- **Structure:** 5-level deep hierarchy (C-Suite → VP → Director → Manager → IC)
- **Org Leader:** Catherine Williams (CEO)

### Hierarchy Structure:
```
Catherine Williams (CEO) [ORG LEADER]
├── Jonathan Martinez (COO)
│   ├── Patricia Kim (VP of Sales - Americas)
│   │   └── Christopher Lee (Director of Sales - North America)
│   │       ├── Grace Hill (Sales Manager - Enterprise)
│   │       │   └── Zoe Garcia (Senior Sales Representative)
│   │       └── Tyler Wang (Sales Manager - SMB)
│   │           └── Ian Miller (Sales Representative)
│   ├── Michael Johnson (VP of Sales - EMEA)
│   │   ├── Thomas Clark (Director of Sales - UK)
│   │   └── Maria Gonzalez (Director of Sales - Germany)
│   └── Sandra Liu (VP of Sales - APAC)
│       ├── Kevin Zhang (Director of Sales - China)
│       └── Rachel Patel (Director of Sales - India)
├── Elizabeth Chen (CTO)
│   ├── David Wilson (VP of Engineering)
│   │   ├── James Murphy (Director of Backend Engineering)
│   │   │   ├── Brian Collins (Senior Engineering Manager)
│   │   │   │   └── Noah White (Senior Software Engineer)
│   │   │   └── Jessica Rodriguez (Engineering Manager - Platform)
│   │   │       └── Emma Clark (Software Engineer)
│   │   ├── Emily Turner (Director of Frontend Engineering)
│   │   │   └── Alex Thompson (Engineering Manager - API)
│   │   │       └── Liam Gonzalez (Frontend Engineer)
│   │   └── Daniel Cooper (Director of Mobile Engineering)
│   │       └── Maya Johnson (Engineering Manager - Mobile)
│   │           └── Ava Zhang (Mobile Developer)
│   ├── Jennifer Brown (VP of Product)
│   │   ├── Ashley Adams (Director of Product Management)
│   │   │   ├── Lucas Davis (Product Manager - Core)
│   │   │   │   └── Owen Patel (Product Analyst)
│   │   │   └── Chloe Wilson (Product Manager - Analytics)
│   │   │       └── Mia Murphy (Data Analyst)
│   │   └── Ryan Scott (Director of UX Design)
│   │       └── Ethan Brown (UX Manager)
│   │           └── Logan Turner (UX Designer)
│   └── Robert Garcia (VP of Infrastructure)
│       ├── Victoria Hall (Director of DevOps)
│       └── Nathan Green (Director of Security)
├── Richard Thompson (CFO)
│   └── Lisa Anderson (VP of Finance)
│       ├── Sophie Parker (Director of Financial Planning)
│       └── Jordan Brooks (Director of Corporate Finance)
├── Angela Davis (CMO)
│   ├── Mark Taylor (VP of Marketing - Digital)
│   └── Amanda Miller (VP of Marketing - Brand)
└── Steven Rodriguez (CHRO)
```

**Characteristics:**
- Complex multi-regional structure (Americas, EMEA, APAC)
- Deep specialization (Backend, Frontend, Mobile, DevOps, Security)
- Clear career progression paths
- Enterprise-scale complexity

---

## 4. **Matrix Organization - InnovateCorp**
**File:** `matrix_innovatecorp_dual.csv`
- **Users:** 25 employees
- **Structure:** Matrix with dual reporting (Functional + Project)
- **Org Leader:** Rebecca Johnson (CEO)

### Hierarchy Structure:
```
Rebecca Johnson (CEO) [ORG LEADER]
├── Marcus Williams (VP of Engineering)
│   └── Jennifer Kim (Engineering Director)
│       ├── Lisa Rodriguez (Senior Engineering Manager) [Alpha Project Lead]
│       │   ├── Maya Patel (Senior Software Engineer)
│       │   ├── Amanda Brown (Software Engineer)
│       │   └── Grace Wang (QA Engineer)
│       └── James Wilson (Engineering Manager) [Beta Project Lead]
│           ├── Thomas Anderson (Senior Software Engineer)
│           ├── Kevin Zhang (Software Engineer)
│           └── Tyler Scott (QA Engineer)
├── Diana Chen (VP of Product)
│   └── Sarah Thompson (Product Director)
│       ├── Emma Garcia (Product Manager - Alpha Project)
│       │   ├── Sophie Turner (UX Designer)
│       │   └── Olivia Martinez (Business Analyst)
│       └── David Park (Product Manager - Beta Project)
│           ├── Ryan Murphy (UX Designer)
│           └── Nathan Brooks (Business Analyst)
└── Robert Martinez (VP of Sales)
    └── Michael Davis (Sales Director)
        ├── Rachel Green (Sales Manager - Enterprise) [Alpha Project]
        │   └── Victoria Adams (Sales Representative)
        └── Alex Johnson (Sales Manager - SMB) [Beta Project]
            └── Jordan Hill (Sales Representative)
```

**Characteristics:**
- Dual reporting relationships (functional manager + project manager)
- Project-based teams (Alpha & Beta)
- Cross-functional collaboration
- Modern agile organizational structure

---

## 5. **Regional Structure - MegaCorp International**
**File:** `regional_megacorp_geographic.csv`
- **Users:** 39 employees
- **Structure:** Geographic hierarchy (Global → Regional → Country → Local)
- **Org Leader:** Alexander Hamilton (Global CEO)

### Hierarchy Structure:
```
Alexander Hamilton (Global CEO) [ORG LEADER]
├── Catherine Rodriguez (President - Americas)
│   ├── Maria Gonzalez (VP Sales - North America)
│   │   ├── Jennifer Williams (Sales Director - USA East)
│   │   │   ├── Robert Thompson (Senior Sales Rep - NYC)
│   │   │   └── Lisa Chen (Senior Sales Rep - Boston)
│   │   ├── Michael Johnson (Sales Director - USA West)
│   │   │   ├── David Park (Sales Rep - San Francisco)
│   │   │   └── Emma Wilson (Sales Rep - Seattle)
│   │   └── Patricia Davis (Sales Director - Canada)
│   │       ├── Sarah Brown (Sales Rep - Toronto)
│   │       └── James Garcia (Sales Rep - Vancouver)
│   └── Carlos Santos (VP Sales - Latin America)
│       ├── Ricardo Silva (Sales Director - Brazil)
│       │   └── Paulo Oliveira (Sales Rep - Rio)
│       └── Ana Martinez (Sales Director - Mexico)
│           └── Diego Lopez (Sales Rep - Guadalajara)
├── Jonathan Schmidt (President - EMEA)
│   ├── Pierre Dubois (VP Sales - Western Europe)
│   │   ├── François Martin (Sales Director - France)
│   │   │   └── Sophie Leblanc (Sales Rep - Lyon)
│   │   └── Isabella Rossi (Sales Director - Italy)
│   │       └── Marco Ferrari (Sales Rep - Rome)
│   ├── Klaus Mueller (Sales Director - Germany)
│   │   └── Hans Weber (Sales Rep - Munich)
│   └── Ahmed Hassan (VP Sales - Middle East)
│       └── Omar Al-Rashid (Sales Director - UAE)
│           └── Fatima Al-Zahra (Sales Rep - Abu Dhabi)
└── Li Wei Chen (President - APAC)
    ├── Raj Sharma (VP Sales - India)
    │   ├── Priya Patel (Sales Manager - Mumbai)
    │   │   └── Vikram Singh (Sales Rep - Mumbai Central)
    │   └── Arjun Kumar (Sales Manager - Bangalore)
    │       └── Ravi Gupta (Sales Rep - Bangalore Tech)
    └── Tanaka Hiroshi (VP Sales - Japan)
        ├── Yuki Nakamura (Sales Manager - Tokyo)
        │   └── Akira Sato (Sales Rep - Tokyo Central)
        └── Kenji Watanabe (Sales Manager - Osaka)
            └── Mei Suzuki (Sales Rep - Osaka Bay)
```

**Characteristics:**
- Multi-continental structure
- Cultural diversity in names and locations
- Geographic responsibility alignment
- Scalable regional model

---

## 6. **Consulting Firm - StrategyPro Partners**
**File:** `consulting_strategypro_project.csv`
- **Users:** 34 employees
- **Structure:** Professional services hierarchy (Partner → Principal → Manager → Consultant → Analyst)
- **Org Leader:** Margaret Thompson (Managing Partner)

### Hierarchy Structure:
```
Margaret Thompson (Managing Partner) [ORG LEADER]
├── Richard Chen (Senior Partner - Strategy)
│   ├── Sarah Kim (Partner - Digital Transformation)
│   │   └── David Johnson (Principal - Corporate Strategy)
│   │       └── Emily White (Senior Manager - Growth Strategy)
│   │           └── Maya Patel (Manager - Business Development)
│   │               └── Sophia Green (Senior Consultant - Market Research)
│   │                   └── Ethan Parker (Consultant - Financial Modeling)
│   │                       └── Aria Bennett (Business Analyst - Market Intelligence)
│   └── Thomas Anderson (Principal - M&A Strategy)
│       └── Christopher Moore (Senior Manager - Market Entry)
│           └── Nathan Scott (Manager - Competitive Analysis)
├── Elizabeth Martinez (Senior Partner - Operations)
│   ├── Michael Davis (Partner - Supply Chain)
│   │   └── Lisa Brown (Principal - Process Optimization)
│   │       └── Daniel Clark (Senior Manager - Operations Excellence)
│   │           └── Jordan Williams (Manager - Process Improvement)
│   │               └── Logan Adams (Senior Consultant - Operational Audit)
│   │                   └── Zoe Cooper (Consultant - Process Mapping)
│   │                       └── Noah Foster (Business Analyst - Workflow Analysis)
│   └── Rachel Miller (Principal - Lean Operations)
│       └── Victoria Jackson (Senior Manager - Supply Chain Analytics)
│           └── Olivia Martinez (Manager - Quality Management)
└── James Rodriguez (Senior Partner - Technology)
    ├── Jennifer Wilson (Partner - IT Strategy)
    │   └── Amanda Garcia (Principal - Cloud Architecture)
    │       └── Ashley Taylor (Senior Manager - Digital Solutions)
    │           └── Grace Zhang (Manager - Software Architecture)
    │               └── Chloe Liu (Senior Consultant - System Integration)
    │                   └── Ian Hughes (Consultant - Database Design)
    │                       └── Emma Watson (Business Analyst - Technical Requirements)
    └── Kevin Lee (Principal - Data Strategy)
        └── Ryan Thompson (Senior Manager - AI Implementation)
            └── Tyler Rodriguez (Manager - Machine Learning)
```

**Characteristics:**
- Professional services hierarchy with clear seniority levels
- Practice area specialization (Strategy, Operations, Technology)
- Pyramid structure typical of consulting firms
- Deep expertise tracks

---

## 7. **Manufacturing Company - ProManufacturing Corp**
**File:** `manufacturing_promanuf_operational.csv`
- **Users:** 40 employees
- **Structure:** Operational hierarchy with shift-based organization
- **Org Leader:** William Harrison (CEO)

### Hierarchy Structure:
```
William Harrison (CEO) [ORG LEADER]
├── Susan Martinez (VP of Operations)
├── Robert Chen (VP of Manufacturing)
│   ├── Linda Thompson (Plant Manager - Detroit)
│   │   ├── Amanda Garcia (Production Supervisor - A Shift)
│   │   │   ├── Ryan Martin (Machine Operator - Line 1)
│   │   │   ├── Maya Wilson (Machine Operator - Line 2)
│   │   │   ├── Lucas Brown (Assembly Worker - Station A)
│   │   │   └── Chloe Wilson (Assembly Worker - Station B)
│   │   ├── Mark Miller (Production Supervisor - B Shift)
│   │   │   ├── Jordan Davis (Machine Operator - Line 3)
│   │   │   └── Ethan Miller (Assembly Worker - Station C)
│   │   └── Sarah Lee (Production Supervisor - C Shift)
│   │       ├── Grace Rodriguez (Machine Operator - Line 1)
│   │       └── Zoe Anderson (Assembly Worker - Station A)
│   ├── Michael Davis (Plant Manager - Chicago)
│   │   ├── Kevin White (Production Supervisor - A Shift)
│   │   │   ├── Tyler Garcia (Machine Operator - Line 1)
│   │   │   ├── Olivia Martinez (Machine Operator - Line 2)
│   │   │   ├── Ian Thompson (Assembly Worker - Station A)
│   │   │   └── Aria White (Assembly Worker - Station B)
│   │   └── Rachel Clark (Production Supervisor - B Shift)
│   │       └── Nathan Lopez (Machine Operator - Line 3)
│   └── Jennifer Wilson (Plant Manager - Atlanta)
│       └── James Taylor (Production Supervisor - A Shift)
│           └── Sophie Johnson (Machine Operator - Line 1)
├── Patricia Kim (VP of Quality)
│   ├── David Johnson (Quality Manager - Detroit)
│   │   ├── Emily Moore (Quality Inspector - Day)
│   │   └── Daniel Jackson (Quality Inspector - Evening)
│   └── Lisa Brown (Quality Manager - Chicago)
│       └── Ashley Thompson (Quality Inspector - Day)
└── Charles Rodriguez (VP of Supply Chain)
    └── Thomas Anderson (Supply Chain Manager)
        ├── Christopher Williams (Warehouse Supervisor)
        │   ├── Noah Garcia (Forklift Operator)
        │   ├── Emma Martinez (Forklift Operator)
        │   ├── Liam Davis (Shipping Clerk)
        │   └── Ava Rodriguez (Receiving Clerk)
        └── Victoria Harris (Inventory Manager)
```

**Characteristics:**
- Multi-plant operations (Detroit, Chicago, Atlanta)
- Shift-based organization (Day, Evening, Night)
- Union vs. Non-Union distinction
- Operational focus with clear production lines

---

## 8. **Financial Services - CapitalFirst Bank**
**File:** `financial_capitalfirst_regulatory.csv`
- **Users:** 35 employees
- **Structure:** Regulatory-compliant hierarchy with licensing requirements
- **Org Leader:** Alexander Mitchell (CEO)

### Hierarchy Structure:
```
Alexander Mitchell (CEO) [ORG LEADER]
├── Catherine Williams (Chief Risk Officer)
│   ├── Jennifer Kim (Senior Risk Manager - Credit)
│   │   ├── Emily Clark (Credit Risk Analyst)
│   │   │   └── Ethan Thompson (Junior Risk Analyst)
│   ├── David Thompson (Senior Risk Manager - Market)
│   │   └── Daniel Taylor (Market Risk Analyst)
│   └── Lisa Wilson (Senior Risk Manager - Operational)
│       └── Ashley Moore (Operational Risk Analyst)
├── Jonathan Davis (Chief Compliance Officer)
│   └── Amanda Brown (Senior Compliance Manager)
│       ├── Christopher Jackson (Compliance Analyst - AML)
│       │   └── Zoe Brown (Junior Compliance Officer)
│       └── Victoria Thompson (Compliance Analyst - Trading)
├── Elizabeth Rodriguez (Chief Investment Officer)
│   └── Thomas Garcia (Senior Portfolio Manager)
│       ├── Ryan Williams (Investment Analyst)
│       │   └── Ian Miller (Research Associate)
│       └── Maya Harris (Portfolio Analyst)
├── Robert Chen (Head of Wealth Management)
│   └── Sarah Miller (Senior Wealth Advisor)
│       ├── Jordan Martin (Wealth Management Associate)
│       │   └── Aria Davis (Client Service Associate)
│       └── Grace Davis (Private Banking Associate)
├── Patricia Martinez (Head of Corporate Banking)
│   └── James Anderson (Senior Relationship Manager)
│       ├── Tyler Rodriguez (Corporate Banking Associate)
│       └── Olivia Garcia (Credit Analyst)
└── Michael Johnson (Head of Retail Banking)
    ├── Rachel Lee (Branch Manager - Downtown)
    │   ├── Nathan Wilson (Senior Teller)
    │   └── Sophie Martinez (Personal Banker)
    └── Kevin White (Branch Manager - Midtown)
        ├── Lucas Johnson (Loan Officer)
        └── Chloe Anderson (Customer Relationship Specialist)
```

**Characteristics:**
- Regulatory compliance focus
- Professional licensing requirements (Series 7, CFA, etc.)
- Risk management emphasis
- Traditional banking hierarchy

---

## Key Testing Scenarios

### Hierarchy Depth Variation:
- **2 Levels:** Startup (flat structure)
- **3 Levels:** SMB (traditional small business)
- **4 Levels:** Matrix, Regional, Consulting
- **5 Levels:** Enterprise, Manufacturing, Financial

### User Count Variation:
- **Small (9-25):** Startup, Matrix
- **Medium (30-40):** SMB, Manufacturing, Regional, Financial
- **Large (50+):** Enterprise, Consulting

### Structural Patterns:
- **Flat:** Startup direct reporting
- **Functional:** SMB departmental structure
- **Geographic:** Regional country-based
- **Matrix:** Dual reporting relationships
- **Professional:** Consulting seniority-based
- **Operational:** Manufacturing shift-based
- **Regulatory:** Financial compliance-driven

### Field Name Variations:
Each CSV uses different field names to test the universal mapper:
- Name variations: "Name", "Full Name", "Employee Name"
- Email variations: "Email", "Work Email", "Corporate Email"
- Manager variations: "Manager Email", "Reports To Email", "Supervisor Email"
- Role variations: "Role", "Job Title", "Position"

This comprehensive test suite ensures the hierarchy processing system can handle real-world organizational complexity and diversity.
