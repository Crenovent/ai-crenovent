"""
Task 3.3.42: "Explain like I'm 5" Simplification Mode Service
- Simplify conversational responses for broader adoption
- LLM prompt style modification for CRO briefing aid
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Explain Like I'm 5 Service")
logger = logging.getLogger(__name__)

class SimplificationLevel(str, Enum):
    TECHNICAL = "technical"          # Original technical language
    BUSINESS = "business"           # Business-friendly terms
    SIMPLE = "simple"              # "Explain like I'm 5" level
    EXECUTIVE = "executive"        # CRO/Executive summary style

class ContentType(str, Enum):
    WORKFLOW_EXPLANATION = "workflow_explanation"
    MODEL_RESULTS = "model_results"
    ERROR_MESSAGE = "error_message"
    TECHNICAL_CONCEPT = "technical_concept"
    BUSINESS_METRICS = "business_metrics"
    COMPLIANCE_REPORT = "compliance_report"

class SimplificationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Content to simplify
    original_text: str
    content_type: ContentType
    current_level: SimplificationLevel = SimplificationLevel.TECHNICAL
    target_level: SimplificationLevel = SimplificationLevel.SIMPLE
    
    # Context
    audience: str = "general_user"  # "cro", "business_user", "general_user"
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SimplificationResponse(BaseModel):
    request_id: str
    simplified_text: str
    simplification_level: SimplificationLevel
    
    # Metadata
    simplification_ratio: float  # How much simpler (0-1)
    key_concepts_preserved: List[str] = Field(default_factory=list)
    analogies_used: List[str] = Field(default_factory=list)
    
    # CRO briefing specific
    executive_summary: Optional[str] = None
    key_takeaways: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SimplificationTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Template details
    content_type: ContentType
    target_level: SimplificationLevel
    
    # Template content
    prompt_template: str
    example_transformations: List[Dict[str, str]] = Field(default_factory=list)
    
    # Rules
    simplification_rules: List[str] = Field(default_factory=list)
    forbidden_terms: List[str] = Field(default_factory=list)
    preferred_analogies: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
simplification_requests_store: Dict[str, SimplificationRequest] = {}
simplification_responses_store: Dict[str, SimplificationResponse] = {}
simplification_templates_store: Dict[str, SimplificationTemplate] = {}

def _create_default_templates():
    """Create default simplification templates"""
    
    # Workflow explanation template
    workflow_template = SimplificationTemplate(
        tenant_id="default",
        content_type=ContentType.WORKFLOW_EXPLANATION,
        target_level=SimplificationLevel.SIMPLE,
        prompt_template="Explain this workflow like you're talking to a 5-year-old: {original_text}",
        example_transformations=[
            {
                "original": "The ML model processes input features through a neural network to generate predictions",
                "simplified": "Think of it like a smart robot that looks at information and makes its best guess about what might happen"
            }
        ],
        simplification_rules=[
            "Use simple words instead of technical jargon",
            "Use analogies and metaphors",
            "Break complex concepts into small steps",
            "Use 'like' and 'imagine' frequently"
        ],
        forbidden_terms=["algorithm", "neural network", "regression", "API"],
        preferred_analogies=["like a smart helper", "imagine a crystal ball", "think of it as a recipe"]
    )
    
    # Model results template
    results_template = SimplificationTemplate(
        tenant_id="default",
        content_type=ContentType.MODEL_RESULTS,
        target_level=SimplificationLevel.SIMPLE,
        prompt_template="Explain these results in the simplest way possible: {original_text}",
        simplification_rules=[
            "Focus on what it means for the business",
            "Use percentages and simple numbers",
            "Explain good vs bad results clearly",
            "Suggest next steps in simple terms"
        ],
        preferred_analogies=["like a report card", "like a weather forecast", "like a health checkup"]
    )
    
    # CRO briefing template
    cro_template = SimplificationTemplate(
        tenant_id="default",
        content_type=ContentType.BUSINESS_METRICS,
        target_level=SimplificationLevel.EXECUTIVE,
        prompt_template="Create an executive summary for CRO: {original_text}",
        simplification_rules=[
            "Start with bottom line impact",
            "Use business metrics and ROI",
            "Highlight risks and opportunities",
            "Provide clear recommendations"
        ],
        preferred_analogies=["revenue impact", "customer experience", "competitive advantage"]
    )
    
    # Store templates
    simplification_templates_store[workflow_template.template_id] = workflow_template
    simplification_templates_store[results_template.template_id] = results_template
    simplification_templates_store[cro_template.template_id] = cro_template
    
    logger.info("Created default simplification templates")

# Initialize templates
_create_default_templates()

@app.post("/simplify", response_model=SimplificationResponse)
async def simplify_content(request: SimplificationRequest):
    """Simplify content to requested level"""
    
    # Store request
    simplification_requests_store[request.request_id] = request
    
    # Find appropriate template
    template = _find_best_template(request.content_type, request.target_level, request.tenant_id)
    
    # Generate simplified content
    simplified_text = _generate_simplified_content(request, template)
    
    # Calculate simplification metrics
    simplification_ratio = _calculate_simplification_ratio(request.original_text, simplified_text)
    key_concepts = _extract_key_concepts(request.original_text)
    analogies = _extract_analogies_used(simplified_text)
    
    # Generate CRO briefing if requested
    executive_summary = None
    key_takeaways = []
    
    if request.audience == "cro" or request.target_level == SimplificationLevel.EXECUTIVE:
        executive_summary = _generate_executive_summary(simplified_text)
        key_takeaways = _generate_key_takeaways(simplified_text)
    
    response = SimplificationResponse(
        request_id=request.request_id,
        simplified_text=simplified_text,
        simplification_level=request.target_level,
        simplification_ratio=simplification_ratio,
        key_concepts_preserved=key_concepts,
        analogies_used=analogies,
        executive_summary=executive_summary,
        key_takeaways=key_takeaways
    )
    
    # Store response
    simplification_responses_store[request.request_id] = response
    
    logger.info(f"Simplified content for {request.target_level.value} level")
    return response

def _find_best_template(content_type: ContentType, target_level: SimplificationLevel, tenant_id: str) -> Optional[SimplificationTemplate]:
    """Find the best matching template"""
    
    # Look for exact match first
    exact_matches = [
        t for t in simplification_templates_store.values()
        if (t.tenant_id == tenant_id and 
            t.content_type == content_type and 
            t.target_level == target_level)
    ]
    
    if exact_matches:
        return exact_matches[0]
    
    # Look for same content type, different level
    content_matches = [
        t for t in simplification_templates_store.values()
        if (t.tenant_id == tenant_id and 
            t.content_type == content_type)
    ]
    
    if content_matches:
        return content_matches[0]
    
    # Look for default templates
    default_matches = [
        t for t in simplification_templates_store.values()
        if (t.tenant_id == "default" and 
            t.content_type == content_type and 
            t.target_level == target_level)
    ]
    
    return default_matches[0] if default_matches else None

def _generate_simplified_content(request: SimplificationRequest, template: Optional[SimplificationTemplate]) -> str:
    """Generate simplified content based on template and rules"""
    
    original_text = request.original_text
    
    if request.target_level == SimplificationLevel.SIMPLE:
        # "Explain like I'm 5" transformations
        simplified = _apply_eli5_transformations(original_text)
    
    elif request.target_level == SimplificationLevel.BUSINESS:
        # Business-friendly transformations
        simplified = _apply_business_transformations(original_text)
    
    elif request.target_level == SimplificationLevel.EXECUTIVE:
        # Executive summary transformations
        simplified = _apply_executive_transformations(original_text)
    
    else:
        # Return original for technical level
        simplified = original_text
    
    return simplified

def _apply_eli5_transformations(text: str) -> str:
    """Apply "Explain like I'm 5" transformations"""
    
    # Dictionary of technical terms to simple explanations
    eli5_replacements = {
        "machine learning model": "smart computer program",
        "algorithm": "set of instructions",
        "neural network": "computer brain",
        "prediction": "best guess",
        "confidence score": "how sure we are",
        "data drift": "when information changes over time",
        "bias": "unfair treatment",
        "API": "way for computers to talk to each other",
        "workflow": "step-by-step process",
        "pipeline": "assembly line for data",
        "regression": "finding patterns",
        "classification": "sorting things into groups",
        "feature": "piece of information",
        "training data": "examples to learn from",
        "validation": "double-checking our work"
    }
    
    simplified = text
    for technical_term, simple_term in eli5_replacements.items():
        simplified = simplified.replace(technical_term, simple_term)
    
    # Add simple analogies
    if "workflow" in text.lower():
        simplified += " Think of it like following a recipe - each step leads to the next until you get the final result."
    
    if "model" in text.lower() and "prediction" in text.lower():
        simplified += " It's like having a crystal ball that looks at patterns to guess what might happen next."
    
    if "data" in text.lower():
        simplified += " Data is just information, like a collection of facts and numbers."
    
    return simplified

def _apply_business_transformations(text: str) -> str:
    """Apply business-friendly transformations"""
    
    business_replacements = {
        "machine learning model": "predictive analytics system",
        "algorithm": "business logic",
        "neural network": "advanced analytics engine",
        "data drift": "changing business conditions",
        "bias": "inconsistent treatment",
        "API": "system integration",
        "pipeline": "automated process",
        "feature": "business metric",
        "training data": "historical business data"
    }
    
    simplified = text
    for technical_term, business_term in business_replacements.items():
        simplified = simplified.replace(technical_term, business_term)
    
    return simplified

def _apply_executive_transformations(text: str) -> str:
    """Apply executive summary transformations"""
    
    # Focus on business impact and outcomes
    if "model" in text.lower() and "prediction" in text.lower():
        return f"Our predictive analytics system helps make better business decisions by analyzing patterns in our data. {text}"
    
    if "workflow" in text.lower():
        return f"This automated business process streamlines operations and improves efficiency. {text}"
    
    return f"Executive Summary: {text}"

def _calculate_simplification_ratio(original: str, simplified: str) -> float:
    """Calculate how much the text was simplified (0-1)"""
    
    # Simple metric based on length and complexity
    original_words = len(original.split())
    simplified_words = len(simplified.split())
    
    if original_words == 0:
        return 0.0
    
    # More words might mean more explanation, less words might mean more concise
    # This is a simplified metric
    length_ratio = min(simplified_words / original_words, 1.0)
    
    # Count technical terms in original vs simplified
    technical_terms = ["algorithm", "neural", "regression", "API", "pipeline", "feature"]
    original_tech_count = sum(1 for term in technical_terms if term in original.lower())
    simplified_tech_count = sum(1 for term in technical_terms if term in simplified.lower())
    
    tech_reduction = 1.0 - (simplified_tech_count / max(original_tech_count, 1))
    
    return (tech_reduction + (1 - length_ratio)) / 2

def _extract_key_concepts(text: str) -> List[str]:
    """Extract key concepts that should be preserved"""
    
    key_concepts = []
    concept_indicators = ["model", "prediction", "data", "workflow", "analysis", "result"]
    
    for concept in concept_indicators:
        if concept in text.lower():
            key_concepts.append(concept)
    
    return key_concepts

def _extract_analogies_used(text: str) -> List[str]:
    """Extract analogies used in simplified text"""
    
    analogies = []
    analogy_indicators = ["like", "think of it as", "imagine", "similar to"]
    
    for indicator in analogy_indicators:
        if indicator in text.lower():
            # Extract the analogy (simplified extraction)
            start_idx = text.lower().find(indicator)
            if start_idx != -1:
                end_idx = text.find(".", start_idx)
                if end_idx != -1:
                    analogies.append(text[start_idx:end_idx])
    
    return analogies

def _generate_executive_summary(text: str) -> str:
    """Generate executive summary for CRO briefing"""
    
    return f"Executive Summary: {text[:200]}..." if len(text) > 200 else f"Executive Summary: {text}"

def _generate_key_takeaways(text: str) -> List[str]:
    """Generate key takeaways for executives"""
    
    takeaways = [
        "System provides automated business insights",
        "Reduces manual decision-making time",
        "Improves accuracy of business predictions"
    ]
    
    # Add specific takeaways based on content
    if "risk" in text.lower():
        takeaways.append("Helps identify and mitigate business risks")
    
    if "customer" in text.lower():
        takeaways.append("Enhances customer experience and retention")
    
    if "revenue" in text.lower() or "sales" in text.lower():
        takeaways.append("Drives revenue growth and sales optimization")
    
    return takeaways[:3]  # Limit to top 3 takeaways

@app.post("/templates", response_model=SimplificationTemplate)
async def create_simplification_template(template: SimplificationTemplate):
    """Create a new simplification template"""
    simplification_templates_store[template.template_id] = template
    logger.info(f"Created simplification template for {template.content_type.value}")
    return template

@app.get("/templates", response_model=List[SimplificationTemplate])
async def get_simplification_templates(
    tenant_id: str,
    content_type: Optional[ContentType] = None,
    target_level: Optional[SimplificationLevel] = None
):
    """Get simplification templates"""
    templates = [t for t in simplification_templates_store.values() if t.tenant_id == tenant_id]
    
    if content_type:
        templates = [t for t in templates if t.content_type == content_type]
    
    if target_level:
        templates = [t for t in templates if t.target_level == target_level]
    
    return templates

@app.get("/simplify/{request_id}", response_model=SimplificationResponse)
async def get_simplification_result(request_id: str):
    """Get simplification result by request ID"""
    
    if request_id not in simplification_responses_store:
        raise HTTPException(status_code=404, detail="Simplification result not found")
    
    return simplification_responses_store[request_id]

@app.get("/analytics/simplification-usage")
async def get_simplification_analytics(tenant_id: str):
    """Get analytics on simplification usage"""
    
    tenant_requests = [
        req for req in simplification_requests_store.values()
        if req.tenant_id == tenant_id
    ]
    
    if not tenant_requests:
        return {"message": "No simplification requests found"}
    
    analytics = {
        "total_requests": len(tenant_requests),
        "by_target_level": {},
        "by_content_type": {},
        "by_audience": {},
        "avg_simplification_ratio": 0
    }
    
    # Analyze by target level
    for level in SimplificationLevel:
        level_requests = [req for req in tenant_requests if req.target_level == level]
        analytics["by_target_level"][level.value] = len(level_requests)
    
    # Analyze by content type
    for content_type in ContentType:
        type_requests = [req for req in tenant_requests if req.content_type == content_type]
        analytics["by_content_type"][content_type.value] = len(type_requests)
    
    # Analyze by audience
    audiences = set(req.audience for req in tenant_requests)
    for audience in audiences:
        audience_requests = [req for req in tenant_requests if req.audience == audience]
        analytics["by_audience"][audience] = len(audience_requests)
    
    # Calculate average simplification ratio
    responses = [
        resp for resp in simplification_responses_store.values()
        if any(req.request_id == resp.request_id and req.tenant_id == tenant_id 
               for req in tenant_requests)
    ]
    
    if responses:
        analytics["avg_simplification_ratio"] = sum(r.simplification_ratio for r in responses) / len(responses)
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Explain Like I'm 5 Service", "task": "3.3.42"}
