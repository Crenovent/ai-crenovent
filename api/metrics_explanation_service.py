"""
Task 4.4.29: Add "Explain metrics" conversational UX (CRO asks: why trust score=0.82?)
- Executive accessibility
- LLM + Metrics API integration
- Conversational mode
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Metrics Explanation Service")
logger = logging.getLogger(__name__)

class ExplanationLevel(str, Enum):
    EXECUTIVE = "executive"    # High-level, business-focused
    TECHNICAL = "technical"    # Detailed, technical explanation
    SIMPLE = "simple"         # Non-technical, easy to understand

class MetricType(str, Enum):
    TRUST_SCORE = "trust_score"
    ACCURACY = "accuracy"
    ROI = "roi"
    ADOPTION = "adoption"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"

class ExplanationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_question: str
    metric_type: Optional[MetricType] = None
    metric_value: Optional[float] = None
    explanation_level: ExplanationLevel = ExplanationLevel.EXECUTIVE
    context: Dict[str, Any] = Field(default_factory=dict)

class ExplanationResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    
    # Main explanation
    explanation: str
    key_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Supporting data
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)
    trend_analysis: Optional[str] = None
    
    # Metadata
    confidence_score: float = 0.8
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
explanation_requests_store: Dict[str, ExplanationRequest] = {}
explanation_responses_store: Dict[str, ExplanationResponse] = {}

class MetricsExplainerLLM:
    def __init__(self):
        # Mock metric data (in production, would integrate with actual metrics services)
        self.mock_metrics = {
            "trust_score": {
                "current_value": 0.82,
                "components": {
                    "accuracy": 0.89,
                    "explainability": 0.78,
                    "drift_stability": 0.85,
                    "bias_fairness": 0.80,
                    "sla_compliance": 0.91
                },
                "trend": "stable",
                "benchmark": 0.85
            },
            "accuracy": {
                "current_value": 89.5,
                "baseline": 85.0,
                "improvement": 4.5,
                "trend": "improving"
            },
            "roi": {
                "current_value": 245.7,
                "cost_savings": 125000,
                "revenue_impact": 89000,
                "trend": "improving"
            }
        }
        
        # Explanation templates
        self.explanation_templates = {
            MetricType.TRUST_SCORE: {
                ExplanationLevel.EXECUTIVE: self._explain_trust_score_executive,
                ExplanationLevel.TECHNICAL: self._explain_trust_score_technical,
                ExplanationLevel.SIMPLE: self._explain_trust_score_simple
            },
            MetricType.ACCURACY: {
                ExplanationLevel.EXECUTIVE: self._explain_accuracy_executive,
                ExplanationLevel.TECHNICAL: self._explain_accuracy_technical,
                ExplanationLevel.SIMPLE: self._explain_accuracy_simple
            },
            MetricType.ROI: {
                ExplanationLevel.EXECUTIVE: self._explain_roi_executive,
                ExplanationLevel.TECHNICAL: self._explain_roi_technical,
                ExplanationLevel.SIMPLE: self._explain_roi_simple
            }
        }
    
    def generate_explanation(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate explanation for a metrics question"""
        
        # Parse the question to determine metric type if not provided
        if not request.metric_type:
            request.metric_type = self._parse_metric_type(request.user_question)
        
        # Get explanation function
        explanation_func = self.explanation_templates.get(
            request.metric_type, {}
        ).get(request.explanation_level)
        
        if not explanation_func:
            return self._generate_generic_explanation(request)
        
        # Generate explanation
        explanation_data = explanation_func(request)
        
        return ExplanationResponse(
            request_id=request.request_id,
            explanation=explanation_data["explanation"],
            key_factors=explanation_data.get("key_factors", []),
            recommendations=explanation_data.get("recommendations", []),
            supporting_metrics=explanation_data.get("supporting_metrics", {}),
            trend_analysis=explanation_data.get("trend_analysis"),
            confidence_score=explanation_data.get("confidence_score", 0.8)
        )
    
    def _parse_metric_type(self, question: str) -> MetricType:
        """Parse question to determine metric type"""
        question_lower = question.lower()
        
        if "trust score" in question_lower or "trust" in question_lower:
            return MetricType.TRUST_SCORE
        elif "accuracy" in question_lower or "accurate" in question_lower:
            return MetricType.ACCURACY
        elif "roi" in question_lower or "return" in question_lower:
            return MetricType.ROI
        elif "adoption" in question_lower or "usage" in question_lower:
            return MetricType.ADOPTION
        elif "performance" in question_lower or "speed" in question_lower:
            return MetricType.PERFORMANCE
        elif "compliance" in question_lower or "regulatory" in question_lower:
            return MetricType.COMPLIANCE
        
        return MetricType.TRUST_SCORE  # Default
    
    def _explain_trust_score_executive(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Executive-level trust score explanation"""
        
        trust_data = self.mock_metrics["trust_score"]
        current_score = trust_data["current_value"]
        
        explanation = f"""Your RBIA trust score is {current_score:.2f} out of 1.0, which indicates high system reliability and trustworthiness. 

This score is calculated from five key components:
• Model Accuracy (89%): Your AI models are performing well above baseline
• Explainability (78%): Decisions are reasonably transparent and interpretable  
• Drift Stability (85%): Models remain stable over time with minimal degradation
• Bias & Fairness (80%): System maintains fair treatment across different groups
• SLA Compliance (91%): System meets operational performance commitments

The score is slightly below our target of 0.85, primarily due to explainability and bias metrics needing attention."""
        
        key_factors = [
            "Strong model accuracy driving overall performance",
            "Excellent SLA compliance showing operational reliability", 
            "Explainability could be improved for better transparency",
            "Bias fairness metrics need monitoring and improvement"
        ]
        
        recommendations = [
            "Focus on improving explainability coverage for newer models",
            "Implement additional bias monitoring for sustained fairness",
            "Continue current practices for accuracy and SLA performance"
        ]
        
        return {
            "explanation": explanation,
            "key_factors": key_factors,
            "recommendations": recommendations,
            "supporting_metrics": trust_data["components"],
            "trend_analysis": "Trust score has been stable over the past quarter",
            "confidence_score": 0.9
        }
    
    def _explain_trust_score_technical(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Technical-level trust score explanation"""
        
        explanation = """Trust score calculation: Weighted average of component metrics.

Components and weights:
- Accuracy (25%): 0.89 → contributes 0.2225
- Explainability (20%): 0.78 → contributes 0.156  
- Drift Stability (20%): 0.85 → contributes 0.17
- Bias Fairness (20%): 0.80 → contributes 0.16
- SLA Compliance (15%): 0.91 → contributes 0.1365

Total: 0.8225 ≈ 0.82

Accuracy is measured via precision/recall on validation sets. Explainability uses SHAP coverage metrics. Drift detection monitors statistical distribution changes. Bias uses demographic parity and equalized odds. SLA tracks latency P95 and uptime."""
        
        return {
            "explanation": explanation,
            "key_factors": ["Weighted scoring algorithm", "Component metric definitions", "Statistical thresholds"],
            "recommendations": ["Increase SHAP coverage", "Add bias monitoring alerts"],
            "confidence_score": 0.95
        }
    
    def _explain_trust_score_simple(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Simple-level trust score explanation"""
        
        explanation = """Your trust score is 0.82 out of 1.0 - that's pretty good! 

Think of it like a report card for your AI system:
• How accurate are the predictions? Grade: B+ (89%)
• Can we explain why decisions were made? Grade: C+ (78%)  
• Does the system stay consistent over time? Grade: B (85%)
• Does it treat everyone fairly? Grade: B- (80%)
• Does it work reliably when you need it? Grade: A- (91%)

Overall: B grade. The system is working well but could be more transparent about its decisions."""
        
        return {
            "explanation": explanation,
            "key_factors": ["Good accuracy", "Reliable performance", "Room for improvement in transparency"],
            "recommendations": ["Make AI decisions more explainable", "Monitor fairness more closely"],
            "confidence_score": 0.85
        }
    
    def _explain_accuracy_executive(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Executive-level accuracy explanation"""
        
        accuracy_data = self.mock_metrics["accuracy"]
        
        explanation = f"""Your RBIA system accuracy is {accuracy_data['current_value']}%, which is {accuracy_data['improvement']} percentage points above your baseline of {accuracy_data['baseline']}%.

This represents strong performance in prediction quality, with the AI making correct decisions nearly 9 out of 10 times. The {accuracy_data['improvement']}% improvement demonstrates clear value from your RBIA investment."""
        
        return {
            "explanation": explanation,
            "key_factors": ["Above-baseline performance", "Consistent improvement trend"],
            "recommendations": ["Maintain current model training practices"],
            "confidence_score": 0.9
        }
    
    def _explain_accuracy_technical(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Technical accuracy explanation"""
        return {
            "explanation": "Accuracy calculated as (TP + TN) / (TP + TN + FP + FN) across validation sets",
            "key_factors": ["Precision/recall balance", "Cross-validation results"],
            "recommendations": ["Review false positive patterns"],
            "confidence_score": 0.95
        }
    
    def _explain_accuracy_simple(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Simple accuracy explanation"""
        return {
            "explanation": "Your AI gets it right 89.5% of the time - that's really good!",
            "key_factors": ["High success rate", "Better than expected"],
            "recommendations": ["Keep doing what you're doing"],
            "confidence_score": 0.9
        }
    
    def _explain_roi_executive(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Executive ROI explanation"""
        
        roi_data = self.mock_metrics["roi"]
        
        explanation = f"""Your RBIA ROI is {roi_data['current_value']}%, meaning you're getting $2.46 back for every dollar invested.

This strong return comes from:
• Cost savings: ${roi_data['cost_savings']:,} per month from automation
• Revenue impact: ${roi_data['revenue_impact']:,} per month from better decisions

The trend is positive, indicating your RBIA investment continues to deliver increasing value."""
        
        return {
            "explanation": explanation,
            "key_factors": ["Strong cost savings", "Positive revenue impact", "Improving trend"],
            "recommendations": ["Consider expanding RBIA to additional workflows"],
            "confidence_score": 0.9
        }
    
    def _explain_roi_technical(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Technical ROI explanation"""
        return {
            "explanation": "ROI = (Gain - Investment) / Investment * 100. Calculated monthly with 12-month projection.",
            "key_factors": ["Time savings valuation", "Revenue attribution model"],
            "recommendations": ["Track additional indirect benefits"],
            "confidence_score": 0.85
        }
    
    def _explain_roi_simple(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Simple ROI explanation"""
        return {
            "explanation": "You're making 2.5x your money back - that's excellent!",
            "key_factors": ["Great return on investment", "Saving money and making money"],
            "recommendations": ["Keep using RBIA - it's working great"],
            "confidence_score": 0.9
        }
    
    def _generate_generic_explanation(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate generic explanation when specific handler not found"""
        
        explanation = f"""I understand you're asking about: "{request.user_question}"

Based on your current RBIA metrics, I can provide insights about system performance, trust scores, accuracy, ROI, and adoption metrics. Could you be more specific about which metric you'd like me to explain?

Available metrics I can explain:
• Trust Score (overall system reliability)
• Accuracy (prediction correctness) 
• ROI (return on investment)
• Adoption (usage and engagement)
• Performance (speed and reliability)
• Compliance (regulatory adherence)"""
        
        return ExplanationResponse(
            request_id=request.request_id,
            explanation=explanation,
            key_factors=["Multiple metrics available", "Need more specific question"],
            recommendations=["Ask about a specific metric for detailed explanation"],
            confidence_score=0.6
        )

# Global explainer instance
explainer = MetricsExplainerLLM()

@app.post("/metrics-explain/ask", response_model=ExplanationResponse)
async def ask_metrics_question(request: ExplanationRequest):
    """Ask a question about metrics and get an explanation"""
    
    # Store request
    explanation_requests_store[request.request_id] = request
    
    # Generate explanation
    response = explainer.generate_explanation(request)
    
    # Store response
    explanation_responses_store[response.response_id] = response
    
    logger.info(f"✅ Generated explanation for question: '{request.user_question}' - Type: {request.metric_type}")
    return response

@app.post("/metrics-explain/quick-ask")
async def quick_ask(
    tenant_id: str,
    question: str,
    explanation_level: ExplanationLevel = ExplanationLevel.EXECUTIVE
):
    """Quick ask without full request model"""
    
    request = ExplanationRequest(
        tenant_id=tenant_id,
        user_question=question,
        explanation_level=explanation_level
    )
    
    return await ask_metrics_question(request)

@app.get("/metrics-explain/conversation/{tenant_id}")
async def get_conversation_history(tenant_id: str, limit: int = 10):
    """Get conversation history for a tenant"""
    
    tenant_requests = [
        req for req in explanation_requests_store.values()
        if req.tenant_id == tenant_id
    ]
    
    # Sort by timestamp (most recent first)
    tenant_requests.sort(key=lambda x: x.request_id, reverse=True)
    tenant_requests = tenant_requests[:limit]
    
    # Get corresponding responses
    conversation = []
    for request in tenant_requests:
        response = next(
            (resp for resp in explanation_responses_store.values() 
             if resp.request_id == request.request_id),
            None
        )
        
        if response:
            conversation.append({
                "question": request.user_question,
                "explanation_level": request.explanation_level,
                "answer": response.explanation,
                "key_factors": response.key_factors,
                "timestamp": response.generated_at.isoformat()
            })
    
    return {
        "tenant_id": tenant_id,
        "conversation_count": len(conversation),
        "conversation": conversation
    }

@app.get("/metrics-explain/popular-questions")
async def get_popular_questions():
    """Get most frequently asked questions"""
    
    # Count question patterns
    question_patterns = {}
    for request in explanation_requests_store.values():
        # Simplified pattern matching
        question_lower = request.user_question.lower()
        
        if "trust score" in question_lower:
            pattern = "trust_score_questions"
        elif "accuracy" in question_lower:
            pattern = "accuracy_questions"
        elif "roi" in question_lower:
            pattern = "roi_questions"
        else:
            pattern = "other_questions"
        
        question_patterns[pattern] = question_patterns.get(pattern, 0) + 1
    
    # Sample popular questions
    popular_questions = [
        "Why is my trust score 0.82?",
        "How is accuracy calculated?",
        "What's driving my ROI of 245%?",
        "Why did my trust score drop this month?",
        "How can I improve my accuracy metrics?"
    ]
    
    return {
        "question_patterns": question_patterns,
        "sample_popular_questions": popular_questions,
        "total_questions": len(explanation_requests_store)
    }

@app.get("/metrics-explain/summary")
async def get_explanation_summary():
    """Get summary of explanation service usage"""
    
    total_requests = len(explanation_requests_store)
    total_responses = len(explanation_responses_store)
    
    # Count by explanation level
    level_counts = {}
    for request in explanation_requests_store.values():
        level = request.explanation_level.value
        level_counts[level] = level_counts.get(level, 0) + 1
    
    # Count by metric type
    metric_counts = {}
    for request in explanation_requests_store.values():
        if request.metric_type:
            metric_type = request.metric_type.value
            metric_counts[metric_type] = metric_counts.get(metric_type, 0) + 1
    
    return {
        "total_requests": total_requests,
        "total_responses": total_responses,
        "requests_by_level": level_counts,
        "requests_by_metric": metric_counts,
        "unique_tenants": len(set(req.tenant_id for req in explanation_requests_store.values()))
    }
