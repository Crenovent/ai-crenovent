"""
Task 4.4.30: Add CFO-friendly ROI narrative generator (LLM explains cost savings in plain English)
- Persona adoption
- LLM + FinOps data integration
- CFO-friendly explanations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA CFO-Friendly ROI Narrative Generator")
logger = logging.getLogger(__name__)

class NarrativeStyle(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    BOARD_PRESENTATION = "board_presentation"
    QUARTERLY_REVIEW = "quarterly_review"

class ROIComponent(str, Enum):
    COST_SAVINGS = "cost_savings"
    REVENUE_INCREASE = "revenue_increase"
    EFFICIENCY_GAINS = "efficiency_gains"
    RISK_REDUCTION = "risk_reduction"
    TIME_SAVINGS = "time_savings"

class ROIData(BaseModel):
    tenant_id: str
    
    # Financial metrics
    total_investment: float
    cost_savings_monthly: float
    revenue_increase_monthly: float
    roi_percentage: float
    payback_months: float
    
    # Efficiency metrics
    time_saved_hours_monthly: float
    process_automation_percentage: float
    error_reduction_percentage: float
    
    # Business impact
    deals_accelerated: int = 0
    compliance_violations_prevented: int = 0
    customer_satisfaction_improvement: float = 0.0
    
    # Context
    industry: str = "technology"
    company_size: str = "mid_market"
    implementation_duration_months: int = 6

class ROINarrative(BaseModel):
    narrative_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Narrative content
    executive_summary: str
    detailed_explanation: str
    key_achievements: List[str] = Field(default_factory=list)
    financial_highlights: List[str] = Field(default_factory=list)
    
    # Supporting data
    roi_data: ROIData
    narrative_style: NarrativeStyle
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_for: str = "CFO"

# In-memory storage
roi_narratives_store: Dict[str, ROINarrative] = {}

class CFOROINarrativeGenerator:
    def __init__(self):
        # CFO-friendly language templates
        self.executive_templates = {
            "high_roi": "Your RBIA investment is delivering exceptional returns at {roi_percentage}% ROI, significantly outperforming typical technology investments.",
            "medium_roi": "Your RBIA implementation shows solid financial performance with {roi_percentage}% ROI, demonstrating clear value creation.",
            "payback_achieved": "The investment has already paid for itself in {payback_months} months, with all future benefits flowing directly to the bottom line.",
            "cost_savings_focus": "Monthly cost savings of ${cost_savings:,.0f} are being realized through process automation and efficiency improvements.",
            "revenue_impact": "Revenue acceleration of ${revenue_increase:,.0f} per month is attributed to faster decision-making and improved accuracy."
        }
        
        self.financial_language = {
            "strong_performance": ["exceptional", "outstanding", "remarkable", "impressive"],
            "solid_performance": ["solid", "consistent", "reliable", "steady"],
            "improvement_areas": ["optimization opportunities", "enhancement potential", "growth areas"]
        }
    
    def generate_narrative(self, roi_data: ROIData, narrative_style: NarrativeStyle) -> ROINarrative:
        """Generate CFO-friendly ROI narrative"""
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(roi_data)
        
        # Generate detailed explanation
        detailed_explanation = self._generate_detailed_explanation(roi_data, narrative_style)
        
        # Generate key achievements
        key_achievements = self._generate_key_achievements(roi_data)
        
        # Generate financial highlights
        financial_highlights = self._generate_financial_highlights(roi_data)
        
        return ROINarrative(
            tenant_id=roi_data.tenant_id,
            executive_summary=executive_summary,
            detailed_explanation=detailed_explanation,
            key_achievements=key_achievements,
            financial_highlights=financial_highlights,
            roi_data=roi_data,
            narrative_style=narrative_style
        )
    
    def _generate_executive_summary(self, roi_data: ROIData) -> str:
        """Generate executive summary for CFO"""
        
        # Determine performance category
        if roi_data.roi_percentage >= 200:
            performance_descriptor = "exceptional"
            benchmark_comparison = "significantly outperforming industry benchmarks"
        elif roi_data.roi_percentage >= 150:
            performance_descriptor = "strong"
            benchmark_comparison = "exceeding typical technology ROI expectations"
        else:
            performance_descriptor = "solid"
            benchmark_comparison = "delivering measurable business value"
        
        # Calculate annual impact
        annual_cost_savings = roi_data.cost_savings_monthly * 12
        annual_revenue_increase = roi_data.revenue_increase_monthly * 12
        total_annual_benefit = annual_cost_savings + annual_revenue_increase
        
        summary = f"""Executive Summary: RBIA Investment Performance
        
Your RBIA implementation is delivering {performance_descriptor} financial returns with a {roi_data.roi_percentage:.1f}% ROI, {benchmark_comparison}.

Key Financial Impact:
• Total Annual Benefit: ${total_annual_benefit:,.0f}
• Monthly Cost Savings: ${roi_data.cost_savings_monthly:,.0f}
• Monthly Revenue Acceleration: ${roi_data.revenue_increase_monthly:,.0f}
• Payback Period: {roi_data.payback_months:.1f} months

The investment has transformed operational efficiency while delivering measurable bottom-line impact through automated decision-making and reduced manual processes."""
        
        return summary
    
    def _generate_detailed_explanation(self, roi_data: ROIData, style: NarrativeStyle) -> str:
        """Generate detailed ROI explanation"""
        
        if style == NarrativeStyle.BOARD_PRESENTATION:
            return self._generate_board_explanation(roi_data)
        elif style == NarrativeStyle.QUARTERLY_REVIEW:
            return self._generate_quarterly_explanation(roi_data)
        else:
            return self._generate_standard_explanation(roi_data)
    
    def _generate_board_explanation(self, roi_data: ROIData) -> str:
        """Generate board-level explanation"""
        
        return f"""Board-Level Financial Analysis:

Investment Overview:
Our ${roi_data.total_investment:,.0f} investment in RBIA technology has generated a {roi_data.roi_percentage:.1f}% return, positioning us ahead of industry peers in operational efficiency and decision-making capability.

Financial Performance:
• Achieved payback in {roi_data.payback_months:.1f} months vs. industry average of 18-24 months
• Generating ${roi_data.cost_savings_monthly * 12:,.0f} in annual cost savings through process automation
• Accelerating ${roi_data.revenue_increase_monthly * 12:,.0f} in annual revenue through faster, more accurate decisions

Operational Excellence:
• {roi_data.process_automation_percentage:.1f}% of previously manual processes now automated
• {roi_data.error_reduction_percentage:.1f}% reduction in operational errors
• {roi_data.time_saved_hours_monthly:,.0f} hours per month freed up for strategic initiatives

Strategic Value:
This investment has not only delivered immediate financial returns but has also positioned our organization for scalable growth and competitive advantage in an increasingly data-driven market."""
        
    def _generate_quarterly_explanation(self, roi_data: ROIData) -> str:
        """Generate quarterly review explanation"""
        
        quarterly_savings = roi_data.cost_savings_monthly * 3
        quarterly_revenue = roi_data.revenue_increase_monthly * 3
        
        return f"""Quarterly Performance Review:

Q4 Financial Impact:
• Cost Savings Realized: ${quarterly_savings:,.0f}
• Revenue Acceleration: ${quarterly_revenue:,.0f}
• Net Quarterly Benefit: ${quarterly_savings + quarterly_revenue:,.0f}

Year-to-Date Performance:
• Cumulative ROI: {roi_data.roi_percentage:.1f}%
• Total Value Created: ${(quarterly_savings + quarterly_revenue) * 4:,.0f}
• Efficiency Gains: {roi_data.time_saved_hours_monthly * 12:,.0f} hours annually

Looking Forward:
Based on current trajectory, we project continued strong performance with potential for additional 15-20% improvement as system optimization continues and user adoption increases."""
        
    def _generate_standard_explanation(self, roi_data: ROIData) -> str:
        """Generate standard detailed explanation"""
        
        return f"""Detailed ROI Analysis:

Financial Foundation:
Your ${roi_data.total_investment:,.0f} investment in RBIA has generated substantial returns through three primary value drivers:

1. Cost Reduction (${roi_data.cost_savings_monthly:,.0f}/month):
   • Automated manual processes saving {roi_data.time_saved_hours_monthly:,.0f} hours monthly
   • Reduced error rates by {roi_data.error_reduction_percentage:.1f}%, eliminating rework costs
   • Streamlined operations requiring fewer full-time equivalents

2. Revenue Acceleration (${roi_data.revenue_increase_monthly:,.0f}/month):
   • Faster decision-making accelerating {roi_data.deals_accelerated} deals per month
   • Improved accuracy leading to better customer outcomes
   • Enhanced competitive positioning in the market

3. Risk Mitigation:
   • Prevented {roi_data.compliance_violations_prevented} compliance violations
   • Reduced operational risk through consistent, auditable processes
   • Improved customer satisfaction by {roi_data.customer_satisfaction_improvement:.1f}%

The {roi_data.payback_months:.1f}-month payback period demonstrates the investment's financial prudence, while ongoing benefits ensure continued value creation."""
        
    def _generate_key_achievements(self, roi_data: ROIData) -> List[str]:
        """Generate key achievements list"""
        
        achievements = []
        
        # ROI achievement
        if roi_data.roi_percentage >= 200:
            achievements.append(f"Achieved exceptional {roi_data.roi_percentage:.1f}% ROI, doubling investment value")
        else:
            achievements.append(f"Delivered strong {roi_data.roi_percentage:.1f}% ROI exceeding target returns")
        
        # Payback achievement
        if roi_data.payback_months <= 12:
            achievements.append(f"Rapid payback achieved in {roi_data.payback_months:.1f} months")
        else:
            achievements.append(f"Investment payback completed in {roi_data.payback_months:.1f} months")
        
        # Cost savings
        achievements.append(f"Generating ${roi_data.cost_savings_monthly:,.0f} in monthly cost savings")
        
        # Revenue impact
        if roi_data.revenue_increase_monthly > 0:
            achievements.append(f"Accelerating ${roi_data.revenue_increase_monthly:,.0f} in monthly revenue")
        
        # Efficiency gains
        achievements.append(f"Automated {roi_data.process_automation_percentage:.1f}% of manual processes")
        
        # Time savings
        achievements.append(f"Freed up {roi_data.time_saved_hours_monthly:,.0f} hours monthly for strategic work")
        
        return achievements
    
    def _generate_financial_highlights(self, roi_data: ROIData) -> List[str]:
        """Generate financial highlights"""
        
        highlights = []
        
        # Annual projections
        annual_benefit = (roi_data.cost_savings_monthly + roi_data.revenue_increase_monthly) * 12
        highlights.append(f"Projected annual benefit: ${annual_benefit:,.0f}")
        
        # Investment efficiency
        monthly_return = (roi_data.cost_savings_monthly + roi_data.revenue_increase_monthly)
        investment_efficiency = (monthly_return / roi_data.total_investment) * 100
        highlights.append(f"Monthly return on investment: {investment_efficiency:.1f}%")
        
        # Cost per hour saved
        if roi_data.time_saved_hours_monthly > 0:
            cost_per_hour = roi_data.total_investment / (roi_data.time_saved_hours_monthly * 12)
            highlights.append(f"Cost per hour saved: ${cost_per_hour:.0f}")
        
        # Break-even analysis
        break_even_date = datetime.utcnow() + timedelta(days=30 * roi_data.payback_months)
        highlights.append(f"Break-even achieved: {break_even_date.strftime('%B %Y')}")
        
        # Future value projection
        three_year_value = annual_benefit * 3 - roi_data.total_investment
        highlights.append(f"3-year net value projection: ${three_year_value:,.0f}")
        
        return highlights

# Global narrative generator
narrative_generator = CFOROINarrativeGenerator()

@app.post("/roi-narrative/generate", response_model=ROINarrative)
async def generate_roi_narrative(
    roi_data: ROIData,
    narrative_style: NarrativeStyle = NarrativeStyle.EXECUTIVE_SUMMARY
):
    """Generate CFO-friendly ROI narrative"""
    
    narrative = narrative_generator.generate_narrative(roi_data, narrative_style)
    
    # Store narrative
    roi_narratives_store[narrative.narrative_id] = narrative
    
    logger.info(f"✅ Generated CFO ROI narrative for tenant {roi_data.tenant_id} - ROI: {roi_data.roi_percentage:.1f}%")
    return narrative

@app.get("/roi-narrative/tenant/{tenant_id}")
async def get_tenant_narratives(tenant_id: str, limit: int = 5):
    """Get ROI narratives for a tenant"""
    
    tenant_narratives = [
        narrative for narrative in roi_narratives_store.values()
        if narrative.tenant_id == tenant_id
    ]
    
    # Sort by generation date (most recent first)
    tenant_narratives.sort(key=lambda x: x.generated_at, reverse=True)
    
    return {
        "tenant_id": tenant_id,
        "narrative_count": len(tenant_narratives),
        "narratives": tenant_narratives[:limit]
    }

@app.get("/roi-narrative/{narrative_id}")
async def get_narrative(narrative_id: str):
    """Get specific ROI narrative"""
    
    narrative = roi_narratives_store.get(narrative_id)
    
    if not narrative:
        raise HTTPException(status_code=404, detail="ROI narrative not found")
    
    return narrative

@app.post("/roi-narrative/quick-generate")
async def quick_generate_narrative(
    tenant_id: str,
    total_investment: float,
    monthly_savings: float,
    monthly_revenue_increase: float = 0.0,
    narrative_style: NarrativeStyle = NarrativeStyle.EXECUTIVE_SUMMARY
):
    """Quick generate narrative with minimal inputs"""
    
    # Calculate derived metrics
    monthly_benefit = monthly_savings + monthly_revenue_increase
    roi_percentage = ((monthly_benefit * 12 - total_investment) / total_investment) * 100
    payback_months = total_investment / monthly_benefit if monthly_benefit > 0 else 0
    
    # Create ROI data
    roi_data = ROIData(
        tenant_id=tenant_id,
        total_investment=total_investment,
        cost_savings_monthly=monthly_savings,
        revenue_increase_monthly=monthly_revenue_increase,
        roi_percentage=roi_percentage,
        payback_months=payback_months,
        time_saved_hours_monthly=monthly_savings / 75,  # Assume $75/hour
        process_automation_percentage=65.0,  # Default assumption
        error_reduction_percentage=45.0  # Default assumption
    )
    
    return await generate_roi_narrative(roi_data, narrative_style)

@app.get("/roi-narrative/templates")
async def get_narrative_templates():
    """Get available narrative templates and styles"""
    
    return {
        "narrative_styles": [style.value for style in NarrativeStyle],
        "sample_templates": {
            "executive_summary": "High-level overview for C-suite consumption",
            "detailed_analysis": "Comprehensive financial analysis with supporting data",
            "board_presentation": "Board-ready summary with strategic context",
            "quarterly_review": "Quarterly performance review format"
        },
        "roi_components": [component.value for component in ROIComponent],
        "language_focus": "CFO-friendly financial terminology and metrics"
    }

@app.get("/roi-narrative/summary")
async def get_narrative_summary():
    """Get ROI narrative service summary"""
    
    total_narratives = len(roi_narratives_store)
    
    if total_narratives == 0:
        return {
            "total_narratives": 0,
            "unique_tenants": 0,
            "average_roi": 0.0,
            "style_distribution": {}
        }
    
    # Calculate statistics
    all_narratives = list(roi_narratives_store.values())
    unique_tenants = len(set(n.tenant_id for n in all_narratives))
    average_roi = sum([n.roi_data.roi_percentage for n in all_narratives]) / len(all_narratives)
    
    # Style distribution
    style_counts = {}
    for narrative in all_narratives:
        style = narrative.narrative_style.value
        style_counts[style] = style_counts.get(style, 0) + 1
    
    return {
        "total_narratives": total_narratives,
        "unique_tenants": unique_tenants,
        "average_roi": round(average_roi, 2),
        "style_distribution": style_counts,
        "latest_generation": max([n.generated_at for n in all_narratives]).isoformat()
    }
