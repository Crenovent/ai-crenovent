#!/usr/bin/env python3
"""
Generate 1000+ synthetic training examples for account planning form extraction
"""
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

class PlanningDataGenerator:
    def __init__(self):
        # Expanded industries for better diversity
        self.industries = [
            "Insurance", "Banking", "Technology", "Healthcare", "Manufacturing", 
            "Retail", "Financial Services", "Education", "Government", "Real Estate",
            "Telecommunications", "Energy", "Transportation", "Media", "Entertainment",
            "Automotive", "Aerospace", "Pharmaceuticals", "Biotechnology", "Agriculture",
            "Construction", "Hospitality", "Food & Beverage", "Textiles", "Mining",
            "Consulting", "Legal Services", "Accounting", "Marketing", "Logistics",
            "E-commerce", "Gaming", "Fitness", "Travel", "Security", "Environmental"
        ]
        
        # Expanded account names with various formats
        self.account_names_base = [
            "GlobalTech", "Premier", "MegaBank", "HealthFirst", "AutoManuf", "RetailGiant",
            "EduTech", "GovServices", "PropertyPlus", "InnovateCorp", "SecureLife", "TechFlow",
            "DataSystems", "CloudWorks", "FinanceHub", "MedCare", "BuildCorp", "FoodChain",
            "TransportCo", "EnergyMax", "MediaFlow", "SportsTech", "TravelTime", "SafeGuard",
            "GreenTech", "AeroSpace", "BioPharma", "AgriCorp", "ConstructMax", "HospitalityPlus",
            "FashionHub", "MiningCorp", "ConsultPro", "LegalEagle", "AccountPlus", "MarketMax",
            "LogisticsPro", "GameDev", "FitnessTech", "SecureNet", "CleanTech", "FastDelivery"
        ]
        
        self.company_suffixes = [
            "Solutions", "Group", "Corp", "Systems", "Inc", "LLC", "Ltd", "Industries",
            "Technologies", "Enterprises", "Holdings", "Partners", "Associates", "Company",
            "International", "Global", "Worldwide", "Services", "Consulting", "Labs"
        ]
        
        # Expanded stakeholder names for diversity
        self.first_names = [
            "Sarah", "Michael", "Emily", "David", "Jennifer", "Robert", "Lisa", "James",
            "Maria", "John", "Ashley", "Christopher", "Jessica", "Daniel", "Amanda", "Matthew",
            "Rachel", "Andrew", "Lauren", "Kevin", "Nicole", "Brian", "Stephanie", "Mark",
            "Samantha", "Jason", "Amy", "Ryan", "Michelle", "Aaron", "Heather", "Jonathan",
            "Melissa", "Brandon", "Kimberly", "William", "Anna", "Joseph", "Karen", "Steven",
            "Patricia", "Benjamin", "Linda", "Alexander", "Susan", "Nicholas", "Rebecca", "Anthony"
        ]
        
        self.last_names = [
            "Johnson", "Chen", "Rodriguez", "Kim", "Smith", "Wilson", "Thompson", "Anderson",
            "Garcia", "Davis", "Brown", "Lee", "Miller", "Martinez", "Jones", "Taylor",
            "Williams", "Jackson", "White", "Harris", "Martin", "Clark", "Lewis", "Walker",
            "Hall", "Allen", "Young", "King", "Wright", "Scott", "Torres", "Nguyen",
            "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Gonzalez", "Carter",
            "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans"
        ]
        
        self.roles = [
            "CEO", "CTO", "CFO", "COO", "VP Sales", "Director", "Manager", 
            "President", "VP Marketing", "Head of Operations", "Chief Strategy Officer"
        ]
        
        self.communication_cadences = [
            "Weekly", "Bi-weekly", "Monthly", "Quarterly", "Daily"
        ]
        
        self.account_tiers = ["Strategic", "Key", "Growth", "Emerging"]
        
        self.regions = ["Americas", "EMEA", "APAC", "North America", "Europe", "Asia-Pacific"]
        
        # Enhanced variation patterns for 100k examples
        self.revenue_patterns = [
            "annual revenue of ${amount:,}", "revenue around ${amount_short}", 
            "they have {amount_short} in revenue", "annual revenue is approximately ${amount:,}",
            "revenue of about {amount_short} annually", "{amount_short} annual revenue",
            "generating ${amount:,} per year", "yearly revenue: ${amount:,}",
            "revenue streams total {amount_short}", "financial performance: {amount_short}",
            "top line revenue {amount_short}", "gross revenue ${amount:,}"
        ]
        
        self.communication_patterns = [
            "communication cadence {cadence}", "we should meet {cadence}",
            "schedule {cadence} check-ins", "communication should be {cadence}",
            "{cadence} touchpoints", "{cadence} meetings", "connect {cadence}",
            "{cadence} reviews", "sync up {cadence}", "{cadence} calls",
            "regular {cadence} updates", "{cadence} status meetings"
        ]
        
        self.activity_verbs = [
            "schedule", "plan", "book", "arrange", "set up", "organize", "coordinate"
        ]
        
        self.meeting_types = [
            "meeting", "review", "session", "call", "discussion", "briefing",
            "workshop", "presentation", "conference", "sync", "check-in", "standup"
        ]

    def generate_company_name(self) -> str:
        """Generate diverse company names"""
        base = random.choice(self.account_names_base)
        suffix = random.choice(self.company_suffixes)
        
        # Sometimes add industry prefix
        if random.random() < 0.3:
            industry_prefix = random.choice(["Tech", "Global", "Premier", "Advanced", "Smart"])
            return f"{industry_prefix} {base} {suffix}"
        
        return f"{base} {suffix}"

    def generate_stakeholder_name(self) -> str:
        """Generate diverse stakeholder names"""
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        return f"{first} {last}"

    def format_revenue_amount(self, amount: int) -> Dict[str, str]:
        """Format revenue in various ways"""
        if amount >= 1_000_000_000:
            short = f"{amount // 1_000_000_000}B"
        elif amount >= 1_000_000:
            short = f"{amount // 1_000_000}M"
        elif amount >= 1_000:
            short = f"{amount // 1_000}K"
        else:
            short = str(amount)
            
        return {
            "amount": amount,
            "amount_short": short,
            "formatted": f"${amount:,}"
        }

    def generate_revenue(self) -> int:
        """Generate realistic revenue amounts"""
        tiers = [
            (1_000_000, 10_000_000),    # 1M-10M
            (10_000_000, 100_000_000),  # 10M-100M  
            (100_000_000, 1_000_000_000), # 100M-1B
            (1_000_000_000, 10_000_000_000) # 1B-10B
        ]
        tier = random.choice(tiers)
        return random.randint(tier[0], tier[1])

    def generate_date(self, start_year=2020, end_year=2026) -> str:
        """Generate random date"""
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        delta = end - start
        random_days = random.randint(0, delta.days)
        random_date = start + timedelta(days=random_days)
        return random_date.strftime("%Y-%m-%d")

    def generate_quarter_date(self, year=2024) -> str:
        """Generate Q1, Q2, Q3, Q4 dates"""
        quarters = {
            "Q1": f"{year}-01-01",
            "Q2": f"{year}-04-01", 
            "Q3": f"{year}-07-01",
            "Q4": f"{year}-10-01"
        }
        quarter = random.choice(list(quarters.keys()))
        return quarters[quarter]

    def generate_activities(self) -> List[Dict]:
        """Generate realistic activities"""
        activity_types = [
            "Quarterly Business Review", "Executive Strategy Session", 
            "Product Demo", "Contract Renewal Meeting", "Stakeholder Check-in",
            "Technical Review", "Performance Assessment", "Strategic Planning",
            "Risk Assessment Meeting", "Compliance Review"
        ]
        
        activities = []
        num_activities = random.randint(1, 4)
        
        for _ in range(num_activities):
            activities.append({
                "activity_title": random.choice(activity_types),
                "planned_date": self.generate_date(2024, 2026),
                "activity_type": "Meeting",
                "description": f"Strategic planning meeting for business objectives"
            })
        
        return activities

    def generate_stakeholders(self) -> List[Dict]:
        """Generate realistic stakeholders"""
        stakeholders = []
        num_stakeholders = random.randint(1, 4)  # Increased variety
        
        for _ in range(num_stakeholders):
            stakeholders.append({
                "name": self.generate_stakeholder_name(),
                "role": random.choice(self.roles),
                "influence_level": random.choice(["High", "Medium", "Low"]),
                "relationship_status": random.choice(["Good", "Excellent", "Fair", "Strong", "Developing"])
            })
        
        return stakeholders

    def generate_natural_language_prompt(self, form_data: Dict) -> str:
        """Generate natural language prompts that describe the form data"""
        
        # Enhanced revenue formatting with better patterns
        revenue = form_data["annual_revenue"]
        revenue_info = self.format_revenue_amount(revenue)
        
        revenue_variations = []
        for pattern in self.revenue_patterns:
            try:
                variation = pattern.format(**revenue_info)
                revenue_variations.append(variation)
            except KeyError:
                continue
        
        # Date formatting variations
        customer_since = form_data["customer_since"]
        year = customer_since.split("-")[0]
        month = customer_since.split("-")[1]
        quarter_map = {"01": "Q1", "04": "Q2", "07": "Q3", "10": "Q4"}
        quarter = quarter_map.get(month, "Q1")
        
        date_variations = [
            f"customer since {quarter} of {year}",
            f"been with us since {year}",
            f"client since {customer_since}",
            f"customer relationship started in {quarter} {year}",
            f"been our client since {quarter} of {year}"
        ]
        
        # Enhanced communication variations
        cadence = form_data["communication_cadence"].lower()
        comm_variations = []
        for pattern in self.communication_patterns:
            variation = pattern.format(cadence=cadence)
            comm_variations.append(variation)
        
        # Activity variations
        activities = form_data.get("planned_activities", [])
        activity_text = ""
        if activities:
            activity = activities[0]
            date_obj = datetime.strptime(activity["planned_date"], "%Y-%m-%d")
            month_name = date_obj.strftime("%B")
            day = date_obj.day
            year = date_obj.year
            
            # Enhanced activity variations
            activity_verb = random.choice(self.activity_verbs)
            meeting_type = random.choice(self.meeting_types)
            
            activity_variations = [
                f"{activity_verb} {activity['activity_title']} on {month_name} {day} {year}",
                f"{activity_verb} {meeting_type} on {day} {month_name} {year}",
                f"plan {activity['activity_title']} for {month_name} {day}, {year}",
                f"book {meeting_type} on {day} {month_name} {year}",
                f"organize {activity['activity_title']} {month_name} {day} {year}",
                f"set up {meeting_type} for {day} {month_name} {year}"
            ]
            activity_text = ", " + random.choice(activity_variations)
        
        # Stakeholder variations
        stakeholders = form_data.get("stakeholders", [])
        stakeholder_text = ""
        if stakeholders:
            stakeholder = stakeholders[0]
            stakeholder_variations = [
                f"primary stakeholder is {stakeholder['name']}, {stakeholder['role']}",
                f"key contact is {stakeholder['name']}, {stakeholder['role']} with {stakeholder['influence_level'].lower()} influence",
                f"working with {stakeholder['name']}, {stakeholder['role']}",
                f"stakeholder {stakeholder['name']}, {stakeholder['role']} has {stakeholder['influence_level'].lower()} influence"
            ]
            stakeholder_text = ", " + random.choice(stakeholder_variations)
        
        # Expanded prompt templates for 100k variety
        prompt_templates = [
            f"Draft an account plan for {form_data['account_id']}, {random.choice(date_variations)}, highlighting {form_data['known_risks'].lower()}, and {form_data['key_opportunities'].lower()}{activity_text}, the {random.choice(revenue_variations)}, {random.choice(comm_variations)}{stakeholder_text}",
            
            f"Create strategic account plan for {form_data['account_id']} in {form_data['industry']}, {random.choice(date_variations)}, focus on {form_data['short_term_goals'].lower()}, {random.choice(revenue_variations)}, {random.choice(comm_variations)}{activity_text}{stakeholder_text}",
            
            f"Account planning for {form_data['account_id']}, {random.choice(date_variations)}, {random.choice(revenue_variations)}, targeting {form_data['revenue_growth_target']}% growth, {random.choice(comm_variations)}, address {form_data['known_risks'].lower()}{activity_text}{stakeholder_text}",
            
            f"Strategic plan needed for {form_data['account_id']} ({form_data['industry']}), {random.choice(date_variations)}, {random.choice(revenue_variations)}, focus on {form_data['long_term_goals'].lower()}, {random.choice(comm_variations)}{activity_text}{stakeholder_text}",
            
            f"Develop account strategy for {form_data['account_id']}, {random.choice(date_variations)}, {random.choice(revenue_variations)}, {random.choice(comm_variations)}, opportunities: {form_data['key_opportunities'].lower()}{activity_text}{stakeholder_text}",
            
            f"Plan for {form_data['account_id']} client, {random.choice(date_variations)}, {random.choice(revenue_variations)}, {random.choice(comm_variations)}, focus areas: {form_data['short_term_goals'].lower()}{activity_text}{stakeholder_text}",
            
            f"Account management plan: {form_data['account_id']}, {random.choice(date_variations)}, industry: {form_data['industry']}, {random.choice(revenue_variations)}, {random.choice(comm_variations)}{activity_text}{stakeholder_text}",
            
            f"Strategic planning request for {form_data['account_id']}, {random.choice(date_variations)}, {random.choice(revenue_variations)}, growth target {form_data['revenue_growth_target']}%, {random.choice(comm_variations)}{activity_text}{stakeholder_text}",
            
            f"Customer success plan for {form_data['account_id']} ({form_data['industry']}), {random.choice(date_variations)}, {random.choice(revenue_variations)}, {random.choice(comm_variations)}, risks: {form_data['known_risks'].lower()}{activity_text}{stakeholder_text}",
            
            f"Business plan for {form_data['account_id']}, {random.choice(date_variations)}, {random.choice(revenue_variations)}, objectives: {form_data['long_term_goals'].lower()}, {random.choice(comm_variations)}{activity_text}{stakeholder_text}"
        ]
        
        return random.choice(prompt_templates)

    def generate_form_data(self) -> Dict[str, Any]:
        """Generate complete form data"""
        industry = random.choice(self.industries)
        account_name = self.generate_company_name()  # Use dynamic name generation
        revenue = self.generate_revenue()
        
        # Generate tier based on revenue
        if revenue > 1_000_000_000:
            tier = "Strategic"
        elif revenue > 100_000_000:
            tier = "Key"
        elif revenue > 10_000_000:
            tier = "Growth"
        else:
            tier = "Emerging"
        
        form_data = {
            "account_id": account_name,
            "plan_name": f"Strategic Account Plan - {account_name}",
            "account_owner": f"{industry} Account Manager",
            "industry": industry,
            "annual_revenue": revenue,
            "account_tier": tier,
            "region_territory": random.choice(self.regions),
            "customer_since": self.generate_quarter_date(random.randint(2020, 2024)),
            "short_term_goals": random.choice([
                "Increase market penetration and drive customer adoption",
                "Expand product usage and improve customer satisfaction",
                "Accelerate digital transformation and platform adoption",
                "Strengthen competitive position and increase market share",
                "Drive revenue growth and improve operational efficiency"
            ]),
            "long_term_goals": random.choice([
                "Establish market leadership and sustainable competitive advantage",
                "Build strategic partnership and long-term value creation",
                "Achieve industry-leading customer satisfaction and retention",
                "Become the preferred technology partner in the region",
                "Transform business operations through digital innovation"
            ]),
            "revenue_growth_target": random.randint(10, 50),
            "product_goals": random.choice([
                "Increase solution adoption, Maximize platform utilization, Drive feature engagement",
                "Expand product footprint, Improve user experience, Accelerate time-to-value", 
                "Drive platform adoption, Enhance integration capabilities, Improve analytics"
            ]),
            "customer_success_metrics": random.choice([
                "User adoption rate >85%, Platform uptime >99.9%, Customer satisfaction >4.5/5",
                "Monthly active users >90%, Support ticket resolution <24hrs, NPS >50",
                "Feature utilization >80%, Training completion >95%, Retention rate >98%"
            ]),
            "key_opportunities": random.choice([
                "Market expansion through new product lines and service offerings",
                "Digital transformation acceleration and cloud migration opportunities", 
                "Cross-selling potential in adjacent business units and departments",
                "Strategic partnership development and ecosystem expansion"
            ]),
            "cross_sell_upsell_potential": random.choice([
                "Advanced analytics platform, Premium support services, Training programs",
                "Enterprise security solutions, API integrations, Custom development",
                "Managed services, Professional consulting, Extended warranty"
            ]),
            "known_risks": random.choice([
                "Competitive pressure from emerging market players and pricing challenges",
                "Economic uncertainty impacting budget allocation and investment decisions",
                "Technology adoption challenges and change management resistance",
                "Regulatory compliance requirements and industry standard changes"
            ]),
            "risk_mitigation_strategies": random.choice([
                "Strengthen value proposition through innovation and customer success programs",
                "Develop flexible pricing models and enhance competitive differentiation",
                "Implement proactive customer engagement and change management support",
                "Establish compliance framework and maintain regulatory alignment"
            ]),
            "communication_cadence": random.choice(self.communication_cadences),
            "stakeholders": self.generate_stakeholders(),
            "planned_activities": self.generate_activities()
        }
        
        return form_data

    def generate_training_example(self) -> Dict[str, Any]:
        """Generate a complete training example"""
        form_data = self.generate_form_data()
        prompt = self.generate_natural_language_prompt(form_data)
        
        return {
            "input": prompt,
            "output": form_data,
            "instruction": "Extract structured account planning information from the given text. Return a JSON object with all relevant fields filled."
        }

    def generate_dataset(self, num_examples: int = 100000) -> List[Dict[str, Any]]:
        """Generate full dataset - optimized for large datasets"""
        dataset = []
        
        print(f"🚀 Generating {num_examples:,} training examples...")
        print("   This may take several minutes for large datasets...")
        
        # Use different reporting intervals for large datasets
        report_interval = max(100, num_examples // 100)  # Report every 1% for large datasets
        
        for i in range(num_examples):
            if i % report_interval == 0:
                percentage = (i / num_examples) * 100
                print(f"   Generated {i:,}/{num_examples:,} examples ({percentage:.1f}%)...")
            
            example = self.generate_training_example()
            dataset.append(example)
        
        print(f"✅ Dataset generation complete! {len(dataset):,} examples created.")
        return dataset

    def save_dataset(self, dataset: List[Dict], filename: str = "planning_training_data.json"):
        """Save dataset to file - optimized for large files"""
        os.makedirs("slm_finetuning", exist_ok=True)
        filepath = f"slm_finetuning/{filename}"
        
        print(f"💾 Saving {len(dataset):,} examples to: {filepath}")
        print("   This may take a moment for large datasets...")
        
        # For large datasets, save without indentation to reduce file size
        indent = 2 if len(dataset) < 10000 else None
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=indent, ensure_ascii=False)
        
        # Calculate file size
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ Dataset saved! File size: {file_size:.1f} MB")
        return filepath
    
    def save_dataset_streaming(self, dataset: List[Dict], filename: str = "planning_training_data.jsonl"):
        """Save very large datasets in JSONL format for memory efficiency"""
        os.makedirs("slm_finetuning", exist_ok=True)
        filepath = f"slm_finetuning/{filename}"
        
        print(f"💾 Streaming {len(dataset):,} examples to: {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, example in enumerate(dataset):
                if i % 10000 == 0:
                    print(f"   Saved {i:,}/{len(dataset):,} examples...")
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ Dataset streamed! File size: {file_size:.1f} MB")
        return filepath

def main():
    import sys
    
    # Get number of examples from command line argument
    num_examples = 100000  # Default to 100k
    if len(sys.argv) > 1:
        try:
            num_examples = int(sys.argv[1])
        except ValueError:
            print("⚠️ Invalid number. Using default: 100,000")
    
    print(f"🎯 TARGET: {num_examples:,} training examples")
    
    generator = PlanningDataGenerator()
    
    # Generate dataset
    dataset = generator.generate_dataset(num_examples)
    
    # Choose saving method based on size
    if num_examples > 50000:
        # Use streaming for very large datasets
        filepath = generator.save_dataset_streaming(dataset, f"planning_training_data_{num_examples}.jsonl")
    else:
        # Regular save for smaller datasets
        filepath = generator.save_dataset(dataset, f"planning_training_data_{num_examples}.json")
    
    # Show sample and statistics
    print("\n📋 Sample Training Example:")
    print("-" * 50)
    sample = dataset[0]
    print(f"Input: {sample['input'][:200]}...")
    print(f"Output Keys: {list(sample['output'].keys())}")
    print(f"Revenue: ${sample['output']['annual_revenue']:,}")
    print(f"Activities: {len(sample['output']['planned_activities'])} planned")
    print(f"Stakeholders: {len(sample['output']['stakeholders'])} people")
    
    # Dataset statistics
    industries = set(example['output']['industry'] for example in dataset[:1000])  # Sample for stats
    companies = set(example['output']['account_id'] for example in dataset[:1000])
    
    print(f"\n📊 Dataset Statistics (sample of 1,000):")
    print("-" * 50)
    print(f"Unique Industries: {len(industries)}")
    print(f"Unique Company Names: {len(companies)}")
    print(f"Total Training Examples: {len(dataset):,}")
    print(f"Estimated Training Time: {len(dataset) // 1000} hours (rough estimate)")
    
    return filepath

if __name__ == "__main__":
    main()
