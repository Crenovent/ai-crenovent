"""
Task 3.3.47: Conversational UX KB Fallback Service
- Fallback to knowledge base answers when ML node fails
- RAG + Knowledge Graph integration for safe degraded mode
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA Conversational KB Fallback Service")
logger = logging.getLogger(__name__)

class FallbackTrigger(str, Enum):
    ML_NODE_FAILURE = "ml_node_failure"
    MODEL_UNAVAILABLE = "model_unavailable"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    TIMEOUT = "timeout"
    ERROR = "error"
    MANUAL_OVERRIDE = "manual_override"

class KnowledgeSource(str, Enum):
    KNOWLEDGE_BASE = "knowledge_base"
    DOCUMENTATION = "documentation"
    FAQ = "faq"
    BEST_PRACTICES = "best_practices"
    TROUBLESHOOTING = "troubleshooting"
    TRAINING_MATERIALS = "training_materials"

class FallbackResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    session_id: str
    
    # Original request
    original_query: str
    ml_node_id: Optional[str] = None
    fallback_trigger: FallbackTrigger
    
    # KB response
    kb_answer: str
    confidence_score: float
    knowledge_sources: List[KnowledgeSource] = Field(default_factory=list)
    
    # Supporting information
    related_articles: List[Dict[str, str]] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    
    # Metadata
    is_fallback_response: bool = True
    safe_mode: bool = True
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class KnowledgeArticle(BaseModel):
    article_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Article content
    title: str
    content: str
    summary: str
    category: KnowledgeSource
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Usage
    view_count: int = 0
    helpful_votes: int = 0
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class FallbackConfiguration(BaseModel):
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Fallback settings
    enable_kb_fallback: bool = True
    confidence_threshold: float = 0.3  # Below this, trigger fallback
    max_response_length: int = 500
    
    # Knowledge sources priority
    source_priority: List[KnowledgeSource] = Field(default_factory=lambda: [
        KnowledgeSource.FAQ,
        KnowledgeSource.KNOWLEDGE_BASE,
        KnowledgeSource.DOCUMENTATION,
        KnowledgeSource.BEST_PRACTICES
    ])
    
    # Safety settings
    safe_mode_only: bool = True
    require_human_review: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
knowledge_articles_store: Dict[str, KnowledgeArticle] = {}
fallback_responses_store: Dict[str, FallbackResponse] = {}
fallback_configs_store: Dict[str, FallbackConfiguration] = {}

def _create_default_knowledge_base():
    """Create default knowledge base articles"""
    
    # FAQ articles
    faq_articles = [
        {
            "title": "How to create a workflow",
            "content": "To create a workflow: 1) Navigate to the workflow builder, 2) Drag nodes from the palette, 3) Connect nodes to define the flow, 4) Configure each node's parameters, 5) Save and test your workflow.",
            "summary": "Step-by-step guide to creating workflows",
            "category": KnowledgeSource.FAQ,
            "tags": ["workflow", "creation", "basics"],
            "keywords": ["create", "workflow", "build", "new"]
        },
        {
            "title": "Understanding model confidence scores",
            "content": "Model confidence scores indicate how certain the AI is about its predictions. Scores range from 0-1, where 1 means very confident. Generally, scores above 0.7 are considered reliable for business decisions.",
            "summary": "Explanation of model confidence scoring",
            "category": KnowledgeSource.KNOWLEDGE_BASE,
            "tags": ["confidence", "models", "predictions"],
            "keywords": ["confidence", "score", "prediction", "reliability"]
        },
        {
            "title": "What to do when a model fails",
            "content": "When a model fails: 1) Check the input data quality, 2) Verify the model is properly trained, 3) Review recent changes to the data pipeline, 4) Consider using fallback rules, 5) Contact support if the issue persists.",
            "summary": "Troubleshooting guide for model failures",
            "category": KnowledgeSource.TROUBLESHOOTING,
            "tags": ["troubleshooting", "model", "failure"],
            "keywords": ["model", "fail", "error", "troubleshoot", "fix"]
        },
        {
            "title": "Best practices for data quality",
            "content": "Ensure high data quality by: 1) Validating data at ingestion, 2) Monitoring for missing values, 3) Checking for outliers, 4) Maintaining consistent formats, 5) Regular data audits.",
            "summary": "Guidelines for maintaining data quality",
            "category": KnowledgeSource.BEST_PRACTICES,
            "tags": ["data quality", "best practices"],
            "keywords": ["data", "quality", "validation", "clean"]
        }
    ]
    
    for article_data in faq_articles:
        article = KnowledgeArticle(
            tenant_id="default",
            title=article_data["title"],
            content=article_data["content"],
            summary=article_data["summary"],
            category=article_data["category"],
            tags=article_data["tags"],
            keywords=article_data["keywords"]
        )
        knowledge_articles_store[article.article_id] = article
    
    # Create default configuration
    default_config = FallbackConfiguration(tenant_id="default")
    fallback_configs_store[default_config.config_id] = default_config
    
    logger.info("Created default knowledge base with {} articles".format(len(faq_articles)))

# Initialize default knowledge base
_create_default_knowledge_base()

@app.post("/fallback/configure", response_model=FallbackConfiguration)
async def configure_kb_fallback(config: FallbackConfiguration):
    """Configure KB fallback settings for a tenant"""
    fallback_configs_store[config.config_id] = config
    logger.info(f"Configured KB fallback for tenant {config.tenant_id}")
    return config

@app.get("/fallback/config/{tenant_id}", response_model=FallbackConfiguration)
async def get_fallback_config(tenant_id: str):
    """Get fallback configuration for a tenant"""
    
    # Find tenant-specific config
    tenant_config = None
    for config in fallback_configs_store.values():
        if config.tenant_id == tenant_id:
            tenant_config = config
            break
    
    # Fallback to default config
    if not tenant_config:
        for config in fallback_configs_store.values():
            if config.tenant_id == "default":
                tenant_config = config
                break
    
    if not tenant_config:
        raise HTTPException(status_code=404, detail="Fallback configuration not found")
    
    return tenant_config

@app.post("/fallback/trigger", response_model=FallbackResponse)
async def trigger_kb_fallback(
    tenant_id: str,
    user_id: str,
    session_id: str,
    original_query: str,
    fallback_trigger: FallbackTrigger,
    ml_node_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
):
    """Trigger KB fallback when ML node fails"""
    
    # Get fallback configuration
    config = await get_fallback_config(tenant_id)
    
    if not config.enable_kb_fallback:
        raise HTTPException(status_code=400, detail="KB fallback is disabled for this tenant")
    
    # Search knowledge base for relevant answers
    relevant_articles = _search_knowledge_base(original_query, tenant_id, config.source_priority)
    
    if not relevant_articles:
        # No relevant articles found
        kb_answer = "I'm sorry, I couldn't find relevant information in the knowledge base. Please contact support for assistance."
        confidence_score = 0.1
        knowledge_sources = []
        related_articles = []
        suggested_actions = ["Contact support", "Try rephrasing your question"]
    else:
        # Generate response from best matching article
        best_article = relevant_articles[0]
        kb_answer = _generate_kb_response(best_article, original_query, config.max_response_length)
        confidence_score = _calculate_kb_confidence(best_article, original_query)
        knowledge_sources = [best_article.category]
        
        # Prepare related articles
        related_articles = [
            {
                "title": article.title,
                "summary": article.summary,
                "article_id": article.article_id
            }
            for article in relevant_articles[:3]
        ]
        
        # Generate suggested actions
        suggested_actions = _generate_suggested_actions(best_article, fallback_trigger)
    
    # Create fallback response
    response = FallbackResponse(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        original_query=original_query,
        ml_node_id=ml_node_id,
        fallback_trigger=fallback_trigger,
        kb_answer=kb_answer,
        confidence_score=confidence_score,
        knowledge_sources=knowledge_sources,
        related_articles=related_articles,
        suggested_actions=suggested_actions,
        safe_mode=config.safe_mode_only
    )
    
    # Store response
    fallback_responses_store[response.response_id] = response
    
    # Update article view count
    if relevant_articles:
        relevant_articles[0].view_count += 1
    
    logger.info(f"KB fallback triggered for query: '{original_query}', trigger: {fallback_trigger.value}")
    return response

def _search_knowledge_base(query: str, tenant_id: str, source_priority: List[KnowledgeSource]) -> List[KnowledgeArticle]:
    """Search knowledge base for relevant articles"""
    
    query_lower = query.lower()
    query_words = query_lower.split()
    
    # Get articles for tenant (including default)
    available_articles = [
        article for article in knowledge_articles_store.values()
        if (article.tenant_id == tenant_id or article.tenant_id == "default") and article.is_active
    ]
    
    # Score articles based on relevance
    scored_articles = []
    
    for article in available_articles:
        score = 0
        
        # Title match (highest weight)
        title_words = article.title.lower().split()
        title_matches = sum(1 for word in query_words if word in title_words)
        score += title_matches * 3
        
        # Keyword match (medium weight)
        keyword_matches = sum(1 for word in query_words if word in [kw.lower() for kw in article.keywords])
        score += keyword_matches * 2
        
        # Content match (lower weight)
        content_words = article.content.lower().split()
        content_matches = sum(1 for word in query_words if word in content_words)
        score += content_matches * 1
        
        # Tag match (medium weight)
        tag_matches = sum(1 for word in query_words if word in [tag.lower() for tag in article.tags])
        score += tag_matches * 2
        
        # Boost score based on source priority
        try:
            priority_index = source_priority.index(article.category)
            priority_boost = len(source_priority) - priority_index
            score += priority_boost * 0.5
        except ValueError:
            pass  # Category not in priority list
        
        if score > 0:
            scored_articles.append((article, score))
    
    # Sort by score (descending) and return top articles
    scored_articles.sort(key=lambda x: x[1], reverse=True)
    return [article for article, score in scored_articles[:5]]

def _generate_kb_response(article: KnowledgeArticle, query: str, max_length: int) -> str:
    """Generate a response based on the knowledge base article"""
    
    # Start with the article summary if it's relevant
    if len(article.summary) <= max_length:
        response = article.summary
    else:
        # Use truncated content
        response = article.content[:max_length - 3] + "..."
    
    # Add context about it being a KB response
    kb_prefix = "Based on our knowledge base: "
    
    if len(kb_prefix + response) <= max_length:
        response = kb_prefix + response
    
    return response

def _calculate_kb_confidence(article: KnowledgeArticle, query: str) -> float:
    """Calculate confidence score for KB response"""
    
    query_words = set(query.lower().split())
    
    # Check overlap with title
    title_words = set(article.title.lower().split())
    title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
    
    # Check overlap with keywords
    keyword_words = set(kw.lower() for kw in article.keywords)
    keyword_overlap = len(query_words.intersection(keyword_words)) / len(query_words) if query_words else 0
    
    # Base confidence on overlaps and article quality indicators
    base_confidence = max(title_overlap, keyword_overlap)
    
    # Boost confidence based on article popularity (view count and votes)
    popularity_boost = min(article.view_count / 100, 0.2)  # Max 0.2 boost
    vote_boost = min(article.helpful_votes / 10, 0.1)  # Max 0.1 boost
    
    confidence = min(base_confidence + popularity_boost + vote_boost, 0.9)  # Cap at 0.9 for KB responses
    
    return confidence

def _generate_suggested_actions(article: KnowledgeArticle, trigger: FallbackTrigger) -> List[str]:
    """Generate suggested actions based on the article and fallback trigger"""
    
    actions = []
    
    # Generic actions based on trigger
    if trigger == FallbackTrigger.ML_NODE_FAILURE:
        actions.append("Check the model status in the dashboard")
        actions.append("Try again in a few minutes")
    elif trigger == FallbackTrigger.CONFIDENCE_TOO_LOW:
        actions.append("Provide more specific information")
        actions.append("Try rephrasing your question")
    elif trigger == FallbackTrigger.TIMEOUT:
        actions.append("Try a simpler query")
        actions.append("Check your network connection")
    
    # Article-specific actions
    if article.category == KnowledgeSource.TROUBLESHOOTING:
        actions.append("Follow the troubleshooting steps provided")
        actions.append("Contact support if the issue persists")
    elif article.category == KnowledgeSource.FAQ:
        actions.append("Review the complete FAQ section")
    elif article.category == KnowledgeSource.BEST_PRACTICES:
        actions.append("Implement the recommended best practices")
    
    # Generic helpful actions
    actions.append("View related articles for more information")
    actions.append("Provide feedback on this response")
    
    return actions[:4]  # Limit to 4 actions

@app.post("/knowledge/articles", response_model=KnowledgeArticle)
async def create_knowledge_article(article: KnowledgeArticle):
    """Create a new knowledge base article"""
    knowledge_articles_store[article.article_id] = article
    logger.info(f"Created knowledge article: {article.title}")
    return article

@app.get("/knowledge/articles", response_model=List[KnowledgeArticle])
async def get_knowledge_articles(
    tenant_id: str,
    category: Optional[KnowledgeSource] = None,
    search_query: Optional[str] = None
):
    """Get knowledge base articles"""
    
    articles = [
        article for article in knowledge_articles_store.values()
        if (article.tenant_id == tenant_id or article.tenant_id == "default") and article.is_active
    ]
    
    if category:
        articles = [a for a in articles if a.category == category]
    
    if search_query:
        # Simple search implementation
        search_words = search_query.lower().split()
        filtered_articles = []
        
        for article in articles:
            if any(word in article.title.lower() or 
                   word in article.content.lower() or
                   word in [kw.lower() for kw in article.keywords]
                   for word in search_words):
                filtered_articles.append(article)
        
        articles = filtered_articles
    
    return articles

@app.get("/knowledge/articles/{article_id}", response_model=KnowledgeArticle)
async def get_knowledge_article(article_id: str):
    """Get a specific knowledge article"""
    
    if article_id not in knowledge_articles_store:
        raise HTTPException(status_code=404, detail="Article not found")
    
    article = knowledge_articles_store[article_id]
    article.view_count += 1  # Increment view count
    
    return article

@app.post("/knowledge/articles/{article_id}/vote")
async def vote_on_article(article_id: str, helpful: bool):
    """Vote on whether an article was helpful"""
    
    if article_id not in knowledge_articles_store:
        raise HTTPException(status_code=404, detail="Article not found")
    
    article = knowledge_articles_store[article_id]
    
    if helpful:
        article.helpful_votes += 1
    
    return {"status": "voted", "helpful": helpful, "total_votes": article.helpful_votes}

@app.get("/fallback/responses/{session_id}")
async def get_session_fallback_responses(session_id: str):
    """Get all fallback responses for a session"""
    
    responses = [
        response for response in fallback_responses_store.values()
        if response.session_id == session_id
    ]
    
    return {"session_id": session_id, "fallback_responses": responses}

@app.get("/analytics/fallback-usage")
async def get_fallback_analytics(tenant_id: str):
    """Get analytics on KB fallback usage"""
    
    tenant_responses = [
        response for response in fallback_responses_store.values()
        if response.tenant_id == tenant_id
    ]
    
    if not tenant_responses:
        return {"message": "No fallback responses found"}
    
    analytics = {
        "total_fallback_responses": len(tenant_responses),
        "by_trigger": {},
        "by_knowledge_source": {},
        "avg_confidence": sum(r.confidence_score for r in tenant_responses) / len(tenant_responses),
        "most_common_queries": [],
        "article_usage": {}
    }
    
    # Analyze by trigger
    for trigger in FallbackTrigger:
        trigger_responses = [r for r in tenant_responses if r.fallback_trigger == trigger]
        analytics["by_trigger"][trigger.value] = len(trigger_responses)
    
    # Analyze by knowledge source
    for source in KnowledgeSource:
        source_responses = [r for r in tenant_responses if source in r.knowledge_sources]
        analytics["by_knowledge_source"][source.value] = len(source_responses)
    
    # Find most common queries (simplified)
    query_counts = {}
    for response in tenant_responses:
        query = response.original_query.lower()
        query_counts[query] = query_counts.get(query, 0) + 1
    
    analytics["most_common_queries"] = sorted(
        query_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]
    
    # Article usage statistics
    tenant_articles = [
        article for article in knowledge_articles_store.values()
        if article.tenant_id == tenant_id or article.tenant_id == "default"
    ]
    
    for article in tenant_articles:
        if article.view_count > 0:
            analytics["article_usage"][article.title] = {
                "views": article.view_count,
                "helpful_votes": article.helpful_votes
            }
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Conversational KB Fallback Service", "task": "3.3.47"}
