"""
Task 3.3.46: Industry Glossary Overlays Service
- Complete KG + Ontology integration for RevOps terms mapping
- Extends existing industry overlay manager with conversational UX support
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Industry Glossary Overlays Service")
logger = logging.getLogger(__name__)

class IndustryType(str, Enum):
    SAAS = "saas"
    BANKING = "banking"
    INSURANCE = "insurance"
    ECOMMERCE = "ecommerce"
    FINANCIAL_SERVICES = "financial_services"
    IT_SERVICES = "it_services"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"

class TermCategory(str, Enum):
    METRICS = "metrics"
    PROCESSES = "processes"
    ROLES = "roles"
    TOOLS = "tools"
    CONCEPTS = "concepts"
    COMPLIANCE = "compliance"
    REVOPS = "revops"

class GlossaryTerm(BaseModel):
    term_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Term details
    term: str
    industry: IndustryType
    category: TermCategory
    
    # Definitions
    short_definition: str
    detailed_definition: str
    
    # Context
    usage_examples: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    related_terms: List[str] = Field(default_factory=list)
    
    # Knowledge Graph connections
    kg_entity_id: Optional[str] = None
    ontology_class: Optional[str] = None
    semantic_relationships: List[Dict[str, str]] = Field(default_factory=list)
    
    # Conversational context
    conversational_triggers: List[str] = Field(default_factory=list)  # Phrases that should trigger this definition
    conversation_context: str = ""  # How to explain in conversation
    
    # Metadata
    confidence_score: float = 1.0
    usage_frequency: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class IndustryGlossaryPack(BaseModel):
    pack_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Pack details
    industry: IndustryType
    pack_name: str
    version: str = "1.0"
    description: str
    
    # Terms
    terms: List[str] = Field(default_factory=list)  # term_ids
    
    # Knowledge Graph integration
    kg_namespace: str = ""
    ontology_file_url: Optional[str] = None
    
    # Conversational settings
    enable_auto_suggestions: bool = True
    context_window_size: int = 5  # Number of previous messages to consider
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class ConversationalGlossaryResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    
    # Query context
    original_query: str
    detected_terms: List[str] = Field(default_factory=list)
    
    # Response
    glossary_explanations: List[Dict[str, Any]] = Field(default_factory=list)
    enhanced_response: str = ""
    
    # Knowledge Graph insights
    related_concepts: List[str] = Field(default_factory=list)
    industry_context: str = ""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
glossary_terms_store: Dict[str, GlossaryTerm] = {}
industry_packs_store: Dict[str, IndustryGlossaryPack] = {}
conversational_responses_store: Dict[str, ConversationalGlossaryResponse] = {}

def _create_default_revops_glossary():
    """Create default RevOps glossary terms"""
    
    # SaaS RevOps terms
    saas_terms = [
        {
            "term": "ARR",
            "short_definition": "Annual Recurring Revenue",
            "detailed_definition": "The predictable revenue that a company expects to receive annually from its subscription customers.",
            "usage_examples": ["Our ARR grew by 40% this quarter", "Track ARR by customer segment"],
            "synonyms": ["Annual Recurring Revenue", "Recurring Revenue"],
            "conversational_triggers": ["annual recurring revenue", "arr", "recurring revenue", "subscription revenue"],
            "conversation_context": "ARR is like your company's yearly subscription income - it's the money you can count on getting each year from customers who pay regularly."
        },
        {
            "term": "MRR",
            "short_definition": "Monthly Recurring Revenue",
            "detailed_definition": "The predictable revenue that a company expects to receive monthly from its subscription customers.",
            "usage_examples": ["MRR increased by $50K this month", "Calculate MRR from new signups"],
            "synonyms": ["Monthly Recurring Revenue"],
            "conversational_triggers": ["monthly recurring revenue", "mrr", "monthly revenue"],
            "conversation_context": "MRR is your monthly subscription income - think of it as the steady money coming in each month from your paying customers."
        },
        {
            "term": "CAC",
            "short_definition": "Customer Acquisition Cost",
            "detailed_definition": "The total cost of acquiring a new customer, including marketing and sales expenses.",
            "usage_examples": ["Our CAC is $500 per customer", "Reduce CAC through better targeting"],
            "synonyms": ["Customer Acquisition Cost", "Acquisition Cost"],
            "conversational_triggers": ["customer acquisition cost", "cac", "cost to acquire", "acquisition cost"],
            "conversation_context": "CAC is how much money you spend to get one new customer - like the total cost of your marketing and sales efforts divided by new customers gained."
        },
        {
            "term": "LTV",
            "short_definition": "Customer Lifetime Value",
            "detailed_definition": "The predicted revenue that a customer will generate during their entire relationship with the company.",
            "usage_examples": ["Customer LTV is $2,000", "Improve LTV through upselling"],
            "synonyms": ["Customer Lifetime Value", "CLV", "Lifetime Value"],
            "conversational_triggers": ["lifetime value", "ltv", "clv", "customer value"],
            "conversation_context": "LTV is the total money a customer will pay you over their entire relationship with your company - from first purchase to when they stop being a customer."
        },
        {
            "term": "Churn Rate",
            "short_definition": "Rate at which customers stop subscribing",
            "detailed_definition": "The percentage of customers who cancel their subscriptions during a given time period.",
            "usage_examples": ["Monthly churn rate is 5%", "Reduce churn through better onboarding"],
            "synonyms": ["Customer Churn", "Attrition Rate"],
            "conversational_triggers": ["churn rate", "churn", "customer loss", "attrition"],
            "conversation_context": "Churn rate is like a leaky bucket - it shows what percentage of your customers are leaving each month. Lower churn means you're keeping customers happy."
        },
        {
            "term": "NDR",
            "short_definition": "Net Dollar Retention",
            "detailed_definition": "Measures the percentage of recurring revenue retained from existing customers, including expansions and contractions.",
            "usage_examples": ["NDR of 110% indicates growth", "Track NDR by customer segment"],
            "synonyms": ["Net Dollar Retention", "Net Revenue Retention"],
            "conversational_triggers": ["net dollar retention", "ndr", "net retention", "revenue retention"],
            "conversation_context": "NDR shows if you're growing revenue from existing customers. Above 100% means you're making more money from the same customers through upsells and expansions."
        }
    ]
    
    # Create SaaS glossary pack
    saas_pack = IndustryGlossaryPack(
        tenant_id="default",
        industry=IndustryType.SAAS,
        pack_name="SaaS RevOps Glossary",
        description="Revenue Operations terminology for SaaS companies",
        kg_namespace="saas_revops",
        enable_auto_suggestions=True
    )
    
    # Create and store terms
    term_ids = []
    for term_data in saas_terms:
        term = GlossaryTerm(
            tenant_id="default",
            term=term_data["term"],
            industry=IndustryType.SAAS,
            category=TermCategory.REVOPS,
            short_definition=term_data["short_definition"],
            detailed_definition=term_data["detailed_definition"],
            usage_examples=term_data["usage_examples"],
            synonyms=term_data["synonyms"],
            conversational_triggers=term_data["conversational_triggers"],
            conversation_context=term_data["conversation_context"],
            kg_entity_id=f"saas_revops:{term_data['term'].lower().replace(' ', '_')}",
            ontology_class="RevOpsMetric"
        )
        
        glossary_terms_store[term.term_id] = term
        term_ids.append(term.term_id)
    
    saas_pack.terms = term_ids
    industry_packs_store[saas_pack.pack_id] = saas_pack
    
    logger.info(f"Created SaaS RevOps glossary with {len(saas_terms)} terms")

# Initialize default glossary
_create_default_revops_glossary()

@app.post("/glossary/terms", response_model=GlossaryTerm)
async def create_glossary_term(term: GlossaryTerm):
    """Create a new glossary term"""
    glossary_terms_store[term.term_id] = term
    logger.info(f"Created glossary term: {term.term}")
    return term

@app.get("/glossary/terms", response_model=List[GlossaryTerm])
async def get_glossary_terms(
    tenant_id: str,
    industry: Optional[IndustryType] = None,
    category: Optional[TermCategory] = None,
    search_query: Optional[str] = None
):
    """Get glossary terms with optional filtering"""
    
    terms = [
        term for term in glossary_terms_store.values()
        if (term.tenant_id == tenant_id or term.tenant_id == "default") and term.is_active
    ]
    
    if industry:
        terms = [t for t in terms if t.industry == industry]
    
    if category:
        terms = [t for t in terms if t.category == category]
    
    if search_query:
        search_lower = search_query.lower()
        terms = [
            t for t in terms
            if (search_lower in t.term.lower() or
                search_lower in t.short_definition.lower() or
                any(search_lower in syn.lower() for syn in t.synonyms))
        ]
    
    return terms

@app.post("/glossary/packs", response_model=IndustryGlossaryPack)
async def create_industry_pack(pack: IndustryGlossaryPack):
    """Create a new industry glossary pack"""
    industry_packs_store[pack.pack_id] = pack
    logger.info(f"Created industry pack: {pack.pack_name}")
    return pack

@app.get("/glossary/packs", response_model=List[IndustryGlossaryPack])
async def get_industry_packs(
    tenant_id: str,
    industry: Optional[IndustryType] = None
):
    """Get industry glossary packs"""
    
    packs = [
        pack for pack in industry_packs_store.values()
        if (pack.tenant_id == tenant_id or pack.tenant_id == "default") and pack.is_active
    ]
    
    if industry:
        packs = [p for p in packs if p.industry == industry]
    
    return packs

@app.post("/glossary/conversational/enhance", response_model=ConversationalGlossaryResponse)
async def enhance_conversational_response(
    tenant_id: str,
    user_id: str,
    query: str,
    industry: IndustryType,
    conversation_history: Optional[List[str]] = None
):
    """Enhance conversational response with industry glossary terms"""
    
    if conversation_history is None:
        conversation_history = []
    
    # Find industry pack
    industry_pack = None
    for pack in industry_packs_store.values():
        if pack.industry == industry and (pack.tenant_id == tenant_id or pack.tenant_id == "default"):
            industry_pack = pack
            break
    
    if not industry_pack:
        raise HTTPException(status_code=404, detail="Industry glossary pack not found")
    
    # Get terms from the pack
    pack_terms = [
        glossary_terms_store[term_id] for term_id in industry_pack.terms
        if term_id in glossary_terms_store
    ]
    
    # Detect terms in the query
    detected_terms = []
    glossary_explanations = []
    
    query_lower = query.lower()
    
    for term in pack_terms:
        # Check if any trigger phrases match
        for trigger in term.conversational_triggers:
            if trigger.lower() in query_lower:
                detected_terms.append(term.term)
                
                explanation = {
                    "term": term.term,
                    "definition": term.conversation_context or term.short_definition,
                    "category": term.category.value,
                    "usage_example": term.usage_examples[0] if term.usage_examples else "",
                    "related_terms": term.related_terms[:3]  # Limit to 3 related terms
                }
                glossary_explanations.append(explanation)
                
                # Update usage frequency
                term.usage_frequency += 1
                break
    
    # Generate enhanced response
    enhanced_response = _generate_enhanced_response(query, glossary_explanations, industry)
    
    # Find related concepts using knowledge graph connections
    related_concepts = _find_related_concepts(detected_terms, pack_terms)
    
    # Generate industry context
    industry_context = _generate_industry_context(industry, detected_terms)
    
    response = ConversationalGlossaryResponse(
        tenant_id=tenant_id,
        user_id=user_id,
        original_query=query,
        detected_terms=detected_terms,
        glossary_explanations=glossary_explanations,
        enhanced_response=enhanced_response,
        related_concepts=related_concepts,
        industry_context=industry_context
    )
    
    conversational_responses_store[response.response_id] = response
    
    logger.info(f"Enhanced conversational response with {len(detected_terms)} glossary terms")
    return response

def _generate_enhanced_response(
    query: str,
    explanations: List[Dict[str, Any]],
    industry: IndustryType
) -> str:
    """Generate enhanced response with glossary explanations"""
    
    if not explanations:
        return f"I understand your {industry.value} question. Let me help you with that."
    
    enhanced = f"Great question about {industry.value}! "
    
    if len(explanations) == 1:
        exp = explanations[0]
        enhanced += f"When you mention '{exp['term']}', {exp['definition']} "
        if exp['usage_example']:
            enhanced += f"For example: {exp['usage_example']}. "
    else:
        enhanced += "I notice you're asking about several key terms:\n"
        for exp in explanations:
            enhanced += f"• **{exp['term']}**: {exp['definition']}\n"
    
    enhanced += "How can I help you analyze or work with this information?"
    
    return enhanced

def _find_related_concepts(detected_terms: List[str], pack_terms: List[GlossaryTerm]) -> List[str]:
    """Find related concepts using knowledge graph connections"""
    
    related = []
    
    for term_name in detected_terms:
        # Find the term object
        term_obj = None
        for term in pack_terms:
            if term.term == term_name:
                term_obj = term
                break
        
        if term_obj:
            # Add related terms
            related.extend(term_obj.related_terms)
            
            # Add terms from semantic relationships
            for relationship in term_obj.semantic_relationships:
                if relationship.get("related_term"):
                    related.append(relationship["related_term"])
    
    # Remove duplicates and limit
    return list(set(related))[:5]

def _generate_industry_context(industry: IndustryType, detected_terms: List[str]) -> str:
    """Generate industry-specific context"""
    
    context_templates = {
        IndustryType.SAAS: "In SaaS businesses, these metrics help track subscription growth and customer health.",
        IndustryType.BANKING: "In banking, these terms relate to risk management and regulatory compliance.",
        IndustryType.ECOMMERCE: "In e-commerce, these metrics focus on customer acquisition and conversion.",
        IndustryType.HEALTHCARE: "In healthcare, these terms often relate to patient outcomes and compliance."
    }
    
    base_context = context_templates.get(industry, f"In {industry.value}, these terms are important for business operations.")
    
    if detected_terms:
        base_context += f" The terms you mentioned ({', '.join(detected_terms)}) are particularly relevant for decision-making in this industry."
    
    return base_context

@app.get("/glossary/terms/{term_id}", response_model=GlossaryTerm)
async def get_glossary_term(term_id: str):
    """Get a specific glossary term"""
    
    if term_id not in glossary_terms_store:
        raise HTTPException(status_code=404, detail="Glossary term not found")
    
    term = glossary_terms_store[term_id]
    term.usage_frequency += 1  # Track access
    
    return term

@app.post("/glossary/terms/{term_id}/kg-link")
async def link_term_to_knowledge_graph(
    term_id: str,
    kg_entity_id: str,
    ontology_class: str,
    semantic_relationships: List[Dict[str, str]]
):
    """Link a glossary term to knowledge graph entities"""
    
    if term_id not in glossary_terms_store:
        raise HTTPException(status_code=404, detail="Glossary term not found")
    
    term = glossary_terms_store[term_id]
    term.kg_entity_id = kg_entity_id
    term.ontology_class = ontology_class
    term.semantic_relationships = semantic_relationships
    term.last_updated = datetime.utcnow()
    
    logger.info(f"Linked term {term.term} to KG entity {kg_entity_id}")
    
    return {"status": "linked", "term_id": term_id, "kg_entity_id": kg_entity_id}

@app.get("/glossary/search/conversational")
async def search_conversational_triggers(
    tenant_id: str,
    query: str,
    industry: Optional[IndustryType] = None
):
    """Search for terms based on conversational triggers"""
    
    terms = [
        term for term in glossary_terms_store.values()
        if (term.tenant_id == tenant_id or term.tenant_id == "default") and term.is_active
    ]
    
    if industry:
        terms = [t for t in terms if t.industry == industry]
    
    query_lower = query.lower()
    matching_terms = []
    
    for term in terms:
        for trigger in term.conversational_triggers:
            if trigger.lower() in query_lower:
                matching_terms.append({
                    "term": term.term,
                    "trigger_matched": trigger,
                    "conversation_context": term.conversation_context,
                    "confidence": _calculate_match_confidence(query_lower, trigger.lower())
                })
                break
    
    # Sort by confidence
    matching_terms.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {"query": query, "matches": matching_terms[:10]}

def _calculate_match_confidence(query: str, trigger: str) -> float:
    """Calculate confidence score for trigger match"""
    
    # Exact match gets highest score
    if trigger == query:
        return 1.0
    
    # Substring match
    if trigger in query:
        return 0.8
    
    # Word overlap
    query_words = set(query.split())
    trigger_words = set(trigger.split())
    
    if query_words and trigger_words:
        overlap = len(query_words.intersection(trigger_words))
        return overlap / len(query_words.union(trigger_words))
    
    return 0.0

@app.get("/glossary/analytics/usage")
async def get_glossary_usage_analytics(tenant_id: str):
    """Get analytics on glossary usage"""
    
    tenant_terms = [
        term for term in glossary_terms_store.values()
        if term.tenant_id == tenant_id or term.tenant_id == "default"
    ]
    
    analytics = {
        "total_terms": len(tenant_terms),
        "by_industry": {},
        "by_category": {},
        "most_used_terms": [],
        "recent_enhancements": len([
            r for r in conversational_responses_store.values()
            if r.tenant_id == tenant_id
        ])
    }
    
    # Analyze by industry
    for industry in IndustryType:
        industry_terms = [t for t in tenant_terms if t.industry == industry]
        analytics["by_industry"][industry.value] = len(industry_terms)
    
    # Analyze by category
    for category in TermCategory:
        category_terms = [t for t in tenant_terms if t.category == category]
        analytics["by_category"][category.value] = len(category_terms)
    
    # Most used terms
    used_terms = [(t.term, t.usage_frequency) for t in tenant_terms if t.usage_frequency > 0]
    used_terms.sort(key=lambda x: x[1], reverse=True)
    analytics["most_used_terms"] = used_terms[:10]
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Industry Glossary Overlays Service", "task": "3.3.46"}
