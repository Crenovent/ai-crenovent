#!/usr/bin/env python3
"""
Clean Intent Parser - Dynamic & YAML-Driven
==========================================

LLM-powered intent parsing without hardcoded workflow descriptions.
Dynamically discovers workflows from YAML and extracts parameters.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class IntentType(Enum):
    WORKFLOW_EXECUTION = "workflow_execution"
    DATA_QUERY = "data_query" 
    STATUS_CHECK = "status_check"
    CAPABILITY_DISCOVERY = "capability_discovery"

class AutomationType(Enum):
    RBA = "RBA"
    RBIA = "RBIA"
    AALA = "AALA"

@dataclass
class ParsedIntent:
    """Clean parsed intent structure"""
    intent_type: IntentType
    confidence: float
    target_automation_type: Optional[AutomationType]
    parameters: Dict[str, Any]
    workflow_category: Optional[str]
    raw_input: str
    llm_reasoning: Optional[str] = None

class CleanIntentParser:
    """
    Clean Intent Parser
    
    Features:
    - Dynamic workflow discovery from YAML
    - No hardcoded workflow descriptions
    - Accurate parameter extraction
    - LLM-powered semantic understanding
    """
    
    def __init__(self, pool_manager=None, workflow_loader=None):
        self.pool_manager = pool_manager
        self.workflow_loader = workflow_loader
        self.logger = logging.getLogger(__name__)
        
        # LLM client for intent parsing
        self.llm_client = None
        if pool_manager:
            try:
                self.llm_client = pool_manager.openai_client
            except Exception as e:
                self.logger.warning(f"⚠️ OpenAI client not available: {e}")
    
    async def parse_intent(self, user_input: str, tenant_id: str = "1300") -> ParsedIntent:
        """
        Parse user intent dynamically
        
        Args:
            user_input: Natural language input
            tenant_id: Tenant identifier
            
        Returns:
            ParsedIntent with extracted information
        """
        try:
            # Get available workflows dynamically from YAML
            available_workflows = self._get_available_workflows()
            
            # Use LLM to extract structured intent
            if self.llm_client:
                structured_intent = await self._extract_structured_intent(
                    user_input, available_workflows
                )
                if structured_intent:
                    return self._convert_to_parsed_intent(structured_intent, user_input)
            
            # Fallback to pattern-based parsing
            return self._fallback_pattern_parsing(user_input)
            
        except Exception as e:
            self.logger.error(f"❌ Intent parsing failed: {e}")
            return self._create_fallback_intent(user_input)
    
    def _get_available_workflows(self) -> List[Dict[str, str]]:
        """Get available workflows dynamically from RBA Agent Registry"""
        try:
            # Import registry dynamically to avoid circular imports
            from dsl.registry.enhanced_capability_registry import EnhancedCapabilityRegistry
            rba_registry = EnhancedCapabilityRegistry()
            
            # Initialize registry if needed
            if not hasattr(rba_registry, '_initialized') or not rba_registry._initialized:
                # For sync context, we'll use the capabilities directly
                pass
            
            # Get all supported analysis types from the registry
            analysis_types = rba_registry.get_supported_analysis_types()
            
            workflows = []
            for analysis_type in analysis_types:
                # Create workflow description from analysis type
                description = self._generate_workflow_description(analysis_type)
                workflows.append({
                    "category": analysis_type,
                    "description": description
                })
            
            self.logger.info(f"🎯 Loaded {len(workflows)} workflows dynamically from agent registry")
            return workflows
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load workflows from registry: {e}")
            # Minimal fallback - let the system discover what it can
            return [
                {"category": "general_analysis", "description": "General pipeline analysis"}
            ]
    
    def _generate_workflow_description(self, analysis_type: str) -> str:
        """Generate human-readable description from analysis type"""
        # Dynamic description generation based on analysis type patterns
        type_descriptions = {
            # Data Quality
            'data_quality': 'Pipeline Data Quality Audit - Identify deals missing close dates, amounts, or owners',
            'missing_fields': 'Missing Fields Analysis - Check for incomplete deal data',
            'duplicate_detection': 'Duplicate Deal Detection - Check duplicate deals across Salesforce/CRM',
            'ownerless_deals': 'Ownerless Deals Detection - Detect ownerless or unassigned deals',
            
            # Activity Analysis
            'activity_tracking': 'Activity Gap Analysis - Highlight deals missing activities/logs',
            'quarter_end_dumping': 'Close Date Standardization - Standardize close dates and flag last-day-of-quarter dumping',
            
            # Risk Analysis
            'sandbagging_detection': 'Sandbagging Detection - Identify sandbagging or inflated deals (low probability but high value)',
            'deal_risk_scoring': 'Deal Risk Scoring - Apply risk scoring to deals and categorize into high/medium/low',
            'deals_at_risk': 'At-Risk Deals Analysis - Check my deals at risk (slipping, no activity, customer disengagement)',
            'stale_deals': 'Pipeline Hygiene Check - Run pipeline hygiene check – deals stuck >60 days in stage',
            
            # Velocity Analysis
            'stage_velocity': 'Stage Velocity Analysis - Review pipeline velocity by stage (days per stage)',
            
            # Performance Analysis
            'pipeline_summary': 'Pipeline Summary - Get my pipeline summary (total open, weighted, committed)',
            'coverage_analysis': 'Coverage Ratio Enforcement - Enforce pipeline coverage ratio (e.g., 3x quota rule)',
            'health_overview': 'Company-wide Health Check - View company-wide pipeline health (coverage, risk, stage distribution)',
            'forecast_alignment': 'Forecast Comparison - Compare rep forecast vs system forecast vs AI prediction'
        }
        
        # Check for exact match
        if analysis_type in type_descriptions:
            return type_descriptions[analysis_type]
        
        # Generate description based on keywords
        words = analysis_type.replace('_', ' ').title()
        if 'audit' in analysis_type or 'quality' in analysis_type:
            return f"{words} - Data quality and compliance analysis"
        elif 'risk' in analysis_type or 'scoring' in analysis_type:
            return f"{words} - Risk assessment and scoring analysis"
        elif 'velocity' in analysis_type or 'stage' in analysis_type:
            return f"{words} - Pipeline velocity and progression analysis"
        elif 'activity' in analysis_type:
            return f"{words} - Deal activity and engagement analysis"
        elif 'coverage' in analysis_type or 'summary' in analysis_type:
            return f"{words} - Pipeline performance and coverage analysis"
        else:
            return f"{words} - Pipeline analysis and optimization"
    
    async def _extract_structured_intent(
        self, 
        user_input: str, 
        available_workflows: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Extract structured intent using LLM"""
        try:
            # Build dynamic prompt with available workflows
            workflow_descriptions = []
            for wf in available_workflows:
                workflow_descriptions.append(f"- {wf['category']}: {wf['description']}")
            
            workflows_text = "\n".join(workflow_descriptions)
            
            prompt = f"""
You are an expert pipeline automation system with DEEP UNDERSTANDING of business requirements.

USER REQUEST: "{user_input}"

AVAILABLE WORKFLOWS:
{workflows_text}

DEEP ANALYSIS REQUIREMENTS:
1. UNDERSTAND THE EXACT BUSINESS NEED:
   - What specific data does the user want to see?
   - What exact conditions must be met?
   - What time periods are mentioned?
   - What filters should be applied?

2. EXTRACT PRECISE CONDITIONS - DYNAMIC PARAMETER EXTRACTION:
   - Time periods: ALWAYS extract EXACT numbers and units from ANY format:
     * "30 days", "9 days", "last 7 days", "past 2 weeks", "3 months", "1 year"
     * "in last 30 days", "missing activities in last 9 days", "for 45 days"
     * DEFAULT to 14 days ONLY if no time period mentioned
   - Activity conditions: ANY activity mention = missing_activities = true
   - Missing data: Extract specific fields mentioned or default to ["close_date", "amount", "owner"]
   - Ownership: ownerless/unassigned = owner_filter = null
   - CRITICAL: PRESERVE exact user numbers - if user says "9 days" use 9, not 14!

3. UNDERSTAND BUSINESS CONTEXT & FLEXIBLE MATCHING:
   - ANY mention of "activities", "calls", "emails", "logs", "missing activities" = activity_tracking_audit
     Examples: "missing activities", "no calls", "deals with no activity", "check activities in last X days"
   - ANY mention of "missing", "data quality", "audit", "close date", "amount", "owner" = missing_fields_audit  
     Examples: "missing data", "audit data", "deals missing close dates", "missing amounts"
   - ANY mention of "duplicate", "duplicates", "similar deals" = duplicate_detection
     Examples: "duplicate deals", "check duplicates", "find similar deals", "duplicate check"
   - ANY mention of "ownerless", "unassigned", "no owner", "without owner" = ownerless_deals_detection
     Examples: "ownerless deals", "unassigned", "deals without owners", "no owner"

4. CRITICAL WORKFLOW MAPPING - FOLLOW EXACTLY:
   - Missing activities/logs → "activity_tracking_audit"
   - Missing data fields → "missing_fields_audit" 
   - Duplicate deals → "duplicate_detection"
   - Ownerless/Unassigned deals → "ownerless_deals_detection" (NOT pipeline_hygiene_stale_deals!)
   - Stale/stuck deals, pipeline hygiene check, deals stuck in stage → "pipeline_hygiene_stale_deals"
   - Risk analysis → "risk_scoring_analysis"
   - Sandbagging, inflated deals, low probability high value → "sandbagging_detection"
   - Deals at risk, slipping, customer disengagement → "deals_at_risk"
   - Quarter-end dumping, close date patterns, last-day-of-quarter → "quarter_end_dumping"
   - Pipeline summary, total open, weighted, committed → "pipeline_summary"
   - Pipeline health, company-wide health, health overview, coverage risk stage → "pipeline_health_overview"

5. PIPELINE HYGIENE SPECIFIC ROUTING:
   - ANY mention of "pipeline hygiene", "stuck >X days", "deals stuck in stage", "stale deals" → "pipeline_hygiene_stale_deals"
   - Examples: "Run pipeline hygiene check", "deals stuck >60 days in stage", "stale deals", "pipeline hygiene"
   - NEVER route pipeline hygiene requests to coaching workflows!

6. SANDBAGGING SPECIFIC ROUTING:
   - ANY mention of "sandbagging", "inflated deals", "low probability but high value", "artificially low probability" → "sandbagging_detection"
   - Examples: "Identify sandbagging deals", "high value low probability", "inflated deals", "artificial probability"
   - Focus on high-value deals with suspiciously low probability scores

7. DEALS AT RISK SPECIFIC ROUTING:
   - ANY mention of "deals at risk", "slipping", "customer disengagement", "disengagement", "losing momentum" → "deals_at_risk"
   - Examples: "Check my deals at risk", "deals that are slipping", "customer disengagement", "losing deals"
   - Focus on active deals that need attention to prevent slippage
   - DIFFERENT from activity_tracking_audit which focuses on data quality/missing logs

8. QUARTER-END DUMPING SPECIFIC ROUTING:
   - ANY mention of "quarter-end", "last-day-of-quarter", "dumping", "close date patterns", "standardize close dates" → "quarter_end_dumping"
   - Examples: "Standardize close dates", "flag last-day-of-quarter dumping", "quarter-end patterns", "close date dumping"
   - Focus on revenue recognition patterns and suspicious close date clustering
   - DIFFERENT from missing_fields_audit which checks for missing data

IMPORTANT: If user mentions "ownerless", "unassigned", "no owner", or "without owner", 
ALWAYS use "ownerless_deals_detection" as workflow_category, NEVER use "pipeline_hygiene_stale_deals"!

Extract and return ONLY a JSON object with PRECISE conditions:
{{
    "intent_type": "workflow_execution",
    "confidence": 0.95,
    "target_automation_type": "RBA",
    "workflow_category": "most_appropriate_workflow_category_from_list_above",
    "parameters": {{
        "time_period": "EXACT extracted time from user input (e.g., '30 days', '9 days', '2 weeks')",
        "time_period_days": convert_to_days_number,
        "activity_threshold_days": same_as_time_period_days_for_activity_workflows,
        "required_fields": ["specific", "fields", "mentioned", "or", "default"],
        "missing_activities": true_if_any_activity_keywords_found,
        "owner_filter": "null_if_ownerless_keywords_found",
        "data_quality_check": true_if_missing_or_audit_keywords_found,
        "duplicate_check": true_if_duplicate_keywords_found,
        "threshold": "EXACT number extracted from user input (e.g., '70', '80%', '3.5x', '100000')",
        "amount_threshold": "dollar amounts mentioned (e.g., '$100K', '$50000', '1M')",
        "percentage_threshold": "percentages mentioned (e.g., '80%', '20%', '70')",
        "multiplier_threshold": "multipliers mentioned (e.g., '3x', '1.5x', '2.5')",
        "coverage_ratio": "coverage ratios mentioned (e.g., '3x quota', '4x coverage')",
        "exact_conditions": "precise description based on user's natural language",
        "business_requirement": "flexible interpretation of user's actual need"
    }},
    "reasoning": "detailed explanation of business need and exact conditions to apply"
}}
"""
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a pipeline automation expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
            
        except Exception as e:
            self.logger.error(f"❌ LLM intent extraction failed: {e}")
            return None
    
    def _convert_to_parsed_intent(self, structured_intent: Dict[str, Any], user_input: str) -> ParsedIntent:
        """Convert LLM response to ParsedIntent object"""
        try:
            return ParsedIntent(
                intent_type=IntentType(structured_intent.get("intent_type", "workflow_execution")),
                confidence=structured_intent.get("confidence", 0.8),
                target_automation_type=AutomationType(structured_intent.get("target_automation_type", "RBA")),
                parameters=structured_intent.get("parameters", {}),
                workflow_category=structured_intent.get("workflow_category"),
                raw_input=user_input,
                llm_reasoning=structured_intent.get("reasoning")
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to convert structured intent: {e}")
            return self._create_fallback_intent(user_input)
    
    def _fallback_pattern_parsing(self, user_input: str) -> ParsedIntent:
        """Fallback pattern-based parsing when LLM fails"""
        user_lower = user_input.lower()
        
        # Pattern matching for workflow categories
        if any(term in user_lower for term in ["stuck", "stale", "pipeline hygiene", "days in stage"]):
            workflow_category = "pipeline_hygiene_stale_deals"
        elif any(term in user_lower for term in ["duplicate", "duplicates"]):
            workflow_category = "duplicate_detection"
        elif any(term in user_lower for term in ["missing", "audit", "data quality"]):
            workflow_category = "missing_fields_audit"
        elif any(term in user_lower for term in ["activity", "calls", "emails", "logs"]):
            workflow_category = "activity_tracking_audit"
        elif any(term in user_lower for term in ["risk", "scoring", "categorize"]):
            workflow_category = "risk_scoring_analysis"
        else:
            workflow_category = "pipeline_hygiene_stale_deals"  # Default
        
        # Extract time periods using regex
        parameters = self._extract_parameters_regex(user_input)
        
        return ParsedIntent(
            intent_type=IntentType.WORKFLOW_EXECUTION,
            confidence=0.7,
            target_automation_type=AutomationType.RBA,
            parameters=parameters,
            workflow_category=workflow_category,
            raw_input=user_input,
            llm_reasoning="Pattern-based fallback parsing"
        )
    
    def _extract_parameters_regex(self, user_input: str) -> Dict[str, Any]:
        """Extract parameters using regex patterns"""
        parameters = {
            "entities": ["deals"],
            "actions": ["check"],
            "filters": {}
        }
        
        # Time period extraction with years support
        time_patterns = [
            (r'(\d+)\s*years?', lambda m: (f"{m.group(1)} years", int(m.group(1)) * 365)),
            (r'(\d+)\s*months?', lambda m: (f"{m.group(1)} months", int(m.group(1)) * 30)),
            (r'(\d+)\s*weeks?', lambda m: (f"{m.group(1)} weeks", int(m.group(1)) * 7)),
            (r'(\d+)\s*days?', lambda m: (f"{m.group(1)} days", int(m.group(1)))),
            (r'>(\d+)\s*years?', lambda m: (f">{m.group(1)} years", int(m.group(1)) * 365)),
            (r'>(\d+)\s*months?', lambda m: (f">{m.group(1)} months", int(m.group(1)) * 30)),
            (r'>(\d+)\s*weeks?', lambda m: (f">{m.group(1)} weeks", int(m.group(1)) * 7)),
            (r'>(\d+)\s*days?', lambda m: (f">{m.group(1)} days", int(m.group(1))))
        ]
        
        for pattern, converter in time_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                time_period, days = converter(match)
                parameters["time_period"] = time_period
                parameters["time_period_days"] = days
                break
        
        return parameters
    
    def _create_fallback_intent(self, user_input: str) -> ParsedIntent:
        """Create fallback intent when all parsing fails"""
        return ParsedIntent(
            intent_type=IntentType.WORKFLOW_EXECUTION,
            confidence=0.5,
            target_automation_type=AutomationType.RBA,
            parameters={"entities": ["deals"], "actions": ["check"]},
            workflow_category="pipeline_hygiene_stale_deals",
            raw_input=user_input,
            llm_reasoning="Fallback intent - parsing failed"
        )
