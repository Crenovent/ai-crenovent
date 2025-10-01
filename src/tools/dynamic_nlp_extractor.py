#!/usr/bin/env python3
"""
Dynamic NLP Extractor using LLM for intelligent form field extraction
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DynamicNLPExtractor:
    """Dynamic NLP extractor using LLM for context-aware field extraction"""
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Complete form field schema for comprehensive extraction
        self.form_schema = {
            # Account Details
            "plan_name": "string - Strategic plan name (be specific, not generic)",
            "account_id": "string - Account name/company name", 
            "account_owner": "string - Account manager/owner name",
            "industry": "string - Industry sector (Technology, Finance, Healthcare, etc.)",
            "annual_revenue": "string - Annual revenue as number (convert K/M/billion to numbers)",
            "account_tier": "string - Enterprise/Key/Growth tier",
            "region_territory": "string - Geographic region/territory",
            "customer_since": "string - Date in YYYY-MM-DD format",
            
            # Goals & Objectives  
            "short_term_goals": "string - Short term objectives (be specific, extract from context)",
            "long_term_goals": "string - Long term objectives and vision",
            "revenue_growth_target": "string - Growth target percentage (number only, no % sign)",
            "product_penetration_goals": "string - Product adoption and penetration goals",
            "customer_success_metrics": "string - Success measurement criteria and KPIs",
            
            # Opportunities & Risks
            "key_opportunities": "string - Key business opportunities and potential",
            "cross_sell_upsell_potential": "string - Cross-sell and upsell opportunities", 
            "known_risks": "string - Known risks, challenges, and concerns",
            "risk_mitigation_strategies": "string - Risk mitigation approaches and strategies",
            
            # Engagement Strategy
            "communication_cadence": "string - Communication frequency and schedule",
            "stakeholders": "array - List of objects with name, role, influence (High/Medium/Low), relationship (Positive/Neutral/Negative)",
            "planned_activities": "array - List of objects with title, date (YYYY-MM-DD), description"
        }

    async def extract_fields_dynamically(self, message: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to dynamically extract form fields from natural language"""
        try:
            if not self.pool_manager or not self.pool_manager.openai_client:
                self.logger.warning("No OpenAI client available, falling back to basic extraction")
                return self._basic_fallback_extraction(message)
            
            # Create dynamic extraction prompt
            extraction_prompt = self._create_extraction_prompt(message)
            
            # Call LLM for intelligent extraction
            response = await self._call_llm_for_extraction(extraction_prompt)
            
            if response:
                extracted_data = self._parse_llm_response(response)
                self.logger.info(f"✅ LLM extracted {len(extracted_data)} fields dynamically")
                return extracted_data
            else:
                self.logger.warning("LLM extraction failed, using fallback")
                return self._basic_fallback_extraction(message)
                
        except Exception as e:
            self.logger.error(f"❌ Dynamic extraction error: {e}")
            return self._basic_fallback_extraction(message)

    def _create_extraction_prompt(self, message: str) -> str:
        """Create a comprehensive extraction prompt for the LLM"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        current_quarter = f"Q{((datetime.now().month-1)//3)+1}"
        
        prompt = f"""You are an expert strategic account planning assistant. Your job is to extract ALL possible information from the user's input and fill as many form fields as possible. Be intelligent, contextual, and comprehensive.

USER INPUT: "{message}"

FORM FIELDS TO EXTRACT:
{json.dumps(self.form_schema, indent=2)}

COMPREHENSIVE EXTRACTION RULES:

1. ACCOUNT DETAILS:
   - Extract company/account names from any mention
   - Infer industry from context (tech companies, financial services, etc.)
   - Convert revenue to numbers: "500K" → "500000", "2M" → "2000000", "1.5 billion" → "1500000000"
   - Determine account tier: Large revenue/enterprise → "Enterprise", Medium → "Key", Small → "Growth"
   - Extract regions: APAC, EMEA, Americas, specific countries/states
   - Parse dates flexibly: "since 2020" → "2020-01-01", "January 2023" → "2023-01-01"

2. GOALS & OBJECTIVES:
   - Extract specific goals from any context: "increase revenue", "expand market", "improve efficiency"
   - ALWAYS generate short_term_goals and long_term_goals even if not explicitly mentioned
   - For missing goals, infer based on company context (Microsoft → cloud expansion, digital transformation)
   - Infer product penetration goals from mentions of adoption, usage, expansion
   - Extract success metrics from KPIs, targets, measurements mentioned
   - Parse percentage targets: "15% growth" → "15" (number only)

3. OPPORTUNITIES & RISKS:
   - Identify opportunities from positive mentions: expansion, growth, new markets
   - Extract risks from negative mentions: competition, challenges, concerns
   - Infer mitigation strategies from solutions, approaches, plans mentioned
   - Find cross-sell/upsell potential from product mentions, expansion plans

4. ENGAGEMENT STRATEGY & METRICS:
   - Extract stakeholder information flexibly: "John is the CTO" → stakeholder object
   - Parse meeting schedules: "monthly reviews", "quarterly check-ins"
   - Create activities from any scheduling mentions: "meeting next week", "call on Friday", "progress check meeting on 31 december 2025"
   - Infer communication patterns from frequency mentions
   - Extract product goals as 'product_goals' (not product_penetration_goals)
   - Extract success metrics as 'success_metrics' (not customer_success_metrics)

5. INTELLIGENT INFERENCE:
   - If account tier not mentioned, infer from revenue: >$10M=Enterprise, $1M-10M=Key, <$1M=Growth
   - If industry not mentioned, infer from company name or context
   - Generate relevant plan names based on account and context
   - Create logical defaults for missing but inferable information

6. SMART DEFAULTS (when relevant context exists):
   - Account owner: Extract from "I am", "my name is", or use "Account Manager" 
   - Communication cadence: Infer from business context (Enterprise=Monthly, Key=Quarterly, etc.)
   - Customer since: Use reasonable default if not specified but context suggests long relationship

CRITICAL REQUIREMENTS:
- Extract EVERY piece of relevant information, even if indirect
- Fill ALL applicable fields - don't leave fields empty if you can infer reasonable values
- Be contextually intelligent - understand business scenarios
- Convert all data to proper formats (dates, numbers, etc.)
- Generate comprehensive, business-relevant content

Current context: Today is {current_date}, Current year: {current_year}, Current quarter: {current_quarter}

Return ONLY a valid JSON object with ALL extracted and inferred fields. Aim for maximum field coverage.

IMPORTANT JSON RULES:
- NO comments in JSON (no // or /* */ comments)
- NO trailing commas
- Use proper JSON format only
- All strings must be in double quotes

Example comprehensive response:
{{
  "plan_name": "Strategic Account Plan - Microsoft - {current_year} {current_quarter}",
  "account_id": "Microsoft Corporation",
  "account_owner": "Senior Account Manager",
  "industry": "Technology",
  "annual_revenue": "50000000",
  "account_tier": "Enterprise",
  "region_territory": "Global",
  "customer_since": "{current_year-2}-01-01",
  "short_term_goals": "Achieve 25% revenue growth in cloud services",
  "long_term_goals": "Become primary cloud provider for enterprise transformation",
  "revenue_growth_target": "25",
  "product_penetration_goals": "Increase Azure adoption by 40% across all business units",
  "customer_success_metrics": "Cloud usage metrics, revenue growth, customer satisfaction scores",
  "key_opportunities": "Digital transformation initiatives, cloud migration projects",
  "cross_sell_upsell_potential": "Security services, advanced analytics, AI/ML platforms",
  "known_risks": "Competitive pressure from AWS, budget constraints in economic uncertainty",
  "risk_mitigation_strategies": "Competitive differentiation, flexible pricing models, value demonstration",
  "communication_cadence": "Monthly strategic reviews with quarterly business reviews",
  "stakeholders": [{{"name": "John Smith", "role": "CTO", "influence": "High", "relationship": "Positive"}}],
  "planned_activities": [
    {{"title": "Quarterly Business Review", "date": "{current_date}", "description": "Strategic planning and performance review"}},
    {{"title": "Progress Check Meeting", "date": "2025-12-31", "description": "Progress check meeting as requested"}}
  ]
}}"""

        return prompt

    async def _call_llm_for_extraction(self, prompt: str) -> Optional[str]:
        """Call the LLM for intelligent extraction"""
        try:
            response = await self.pool_manager.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert strategic account planning assistant with deep business intelligence. Extract maximum information from user input and fill ALL possible form fields. Always return valid JSON only with comprehensive data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Slightly higher for more creative inference
                max_tokens=2000  # More tokens for comprehensive extraction
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"❌ LLM call failed: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Clean up response - remove any markdown or extra text
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            if response.startswith("```"):
                response = response[3:]
            
            # Remove JSON comments (// comments) which are invalid in JSON
            import re
            # Remove single-line comments like // comment
            response = re.sub(r'//.*?(?=\n|$)', '', response)
            # Remove multi-line comments like /* comment */
            response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
            # Clean up any trailing commas before closing brackets/braces
            response = re.sub(r',(\s*[}\]])', r'\1', response)
            
            # Parse JSON
            extracted_data = json.loads(response)
            
            # Validate and clean the data
            validated_data = self._validate_extracted_data(extracted_data)
            
            return validated_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Invalid JSON from LLM: {e}")
            self.logger.error(f"Raw response: {response}")
            return {}
        except Exception as e:
            self.logger.error(f"❌ Error parsing LLM response: {e}")
            return {}

    def _validate_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted data"""
        validated = {}
        
        for key, value in data.items():
            if key in self.form_schema and value is not None:
                # Clean up string values
                if isinstance(value, str):
                    value = value.strip()
                    if value:  # Only add non-empty strings
                        validated[key] = value
                elif isinstance(value, (list, dict)):
                    # Validate arrays/objects
                    if value:  # Only add non-empty collections
                        validated[key] = value
                elif isinstance(value, (int, float)):
                    validated[key] = str(value)
        
        return validated

    def _basic_fallback_extraction(self, message: str) -> Dict[str, Any]:
        """Basic fallback extraction for when LLM is not available"""
        import re
        form_data = {}
        message_lower = message.lower()
        
        # Basic stakeholder extraction
        if "stakeholder is" in message_lower:
            # Extract name after "stakeholder is"
            match = re.search(r'stakeholder is\s+([A-Za-z\s]+)', message_lower, re.IGNORECASE)
            if match:
                name = match.group(1).strip().title()
                form_data['stakeholders'] = [{
                    "name": name,
                    "role": "Stakeholder",
                    "influence": "Medium", 
                    "relationship": "Neutral"
                }]
        
        # Basic goal extraction
        if "goal is" in message_lower:
            match = re.search(r'goal is\s+(.+?)(?:\s*,|\s*\.|$)', message_lower, re.IGNORECASE)
            if match:
                goal = match.group(1).strip()
                form_data['short_term_goals'] = goal
        
        return form_data

def get_dynamic_nlp_extractor(pool_manager) -> DynamicNLPExtractor:
    """Returns a singleton instance of the DynamicNLPExtractor"""
    if not hasattr(get_dynamic_nlp_extractor, "instance"):
        get_dynamic_nlp_extractor.instance = DynamicNLPExtractor(pool_manager)
    return get_dynamic_nlp_extractor.instance
