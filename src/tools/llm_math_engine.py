"""
LLM Math Engine for Complex Business Calculations
Advanced mathematical reasoning for strategic planning
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class LLMMathEngine:
    """Advanced mathematical reasoning engine for business calculations"""
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
    
    async def calculate_revenue_projections(self, base_revenue: float, growth_rate: float, 
                                          periods: int = 4, seasonality: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate sophisticated revenue projections with seasonality and risk adjustments
        """
        try:
            if seasonality is None:
                seasonality = {"Q1": 0.9, "Q2": 1.1, "Q3": 1.2, "Q4": 1.0}
            
            calculation_prompt = f"""
Calculate detailed revenue projections with the following parameters:
- Base Revenue: ${base_revenue:,.2f}
- Growth Rate: {growth_rate}%
- Periods: {periods} quarters
- Seasonality Factors: {seasonality}

Perform these calculations:
1. Quarterly revenue projections with seasonality
2. Risk-adjusted projections (90% confidence)
3. Conservative and aggressive scenarios
4. Cumulative revenue over the period
5. Growth trajectory analysis

Provide detailed mathematical reasoning for each calculation.
"""
            
            math_result = await self._execute_math_calculation(calculation_prompt)
            
            # Also calculate using direct math for verification
            projections = []
            cumulative = 0
            
            for i in range(periods):
                quarter = f"Q{(i % 4) + 1}"
                seasonal_factor = seasonality.get(quarter, 1.0)
                period_revenue = base_revenue * (1 + growth_rate/100) ** (i/4) * seasonal_factor
                risk_adjusted = period_revenue * 0.9  # 90% confidence
                
                projection = {
                    "period": quarter,
                    "projected_revenue": round(period_revenue, 2),
                    "risk_adjusted": round(risk_adjusted, 2),
                    "seasonal_factor": seasonal_factor,
                    "growth_factor": round((1 + growth_rate/100) ** (i/4), 3)
                }
                projections.append(projection)
                cumulative += period_revenue
            
            result = {
                "base_parameters": {
                    "base_revenue": base_revenue,
                    "growth_rate": growth_rate,
                    "periods": periods
                },
                "projections": projections,
                "summary": {
                    "total_projected": round(cumulative, 2),
                    "average_quarterly": round(cumulative / periods, 2),
                    "final_quarter": projections[-1]["projected_revenue"] if projections else 0
                },
                "llm_analysis": math_result,
                "calculation_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Revenue projections calculated for {periods} periods")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Revenue projection calculation failed: {e}")
            return {"error": str(e), "projections": []}
    
    async def monte_carlo_risk_analysis(self, base_value: float, volatility: float = 0.2, 
                                      simulations: int = 1000) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation for risk analysis
        """
        try:
            calculation_prompt = f"""
Perform a Monte Carlo risk analysis with these parameters:
- Base Value: ${base_value:,.2f}
- Volatility: {volatility} ({volatility*100}%)
- Simulations: {simulations}

Calculate:
1. Expected value and standard deviation
2. Value at Risk (VaR) at 95% and 99% confidence levels
3. Risk-return ratios
4. Probability distributions
5. Scenario analysis (best case, worst case, most likely)

Show the mathematical methodology and statistical interpretation.
"""
            
            llm_analysis = await self._execute_math_calculation(calculation_prompt)
            
            # Direct Monte Carlo simulation
            import random
            random.seed(42)  # For reproducible results
            
            results = []
            for _ in range(simulations):
                # Log-normal distribution for financial modeling
                shock = random.normalvariate(0, volatility)
                simulated_value = base_value * math.exp(shock)
                results.append(simulated_value)
            
            results.sort()
            
            # Calculate statistics
            mean_value = sum(results) / len(results)
            variance = sum((x - mean_value) ** 2 for x in results) / len(results)
            std_dev = math.sqrt(variance)
            
            # Value at Risk calculations
            var_95_index = int(0.05 * simulations)
            var_99_index = int(0.01 * simulations)
            
            analysis = {
                "simulation_parameters": {
                    "base_value": base_value,
                    "volatility": volatility,
                    "simulations": simulations
                },
                "statistics": {
                    "mean": round(mean_value, 2),
                    "standard_deviation": round(std_dev, 2),
                    "minimum": round(min(results), 2),
                    "maximum": round(max(results), 2),
                    "median": round(results[len(results)//2], 2)
                },
                "risk_metrics": {
                    "var_95_percent": round(results[var_95_index], 2),
                    "var_99_percent": round(results[var_99_index], 2),
                    "expected_shortfall_95": round(sum(results[:var_95_index]) / var_95_index, 2) if var_95_index > 0 else 0,
                    "coefficient_of_variation": round(std_dev / mean_value, 3) if mean_value != 0 else 0
                },
                "scenarios": {
                    "optimistic_10th_percentile": round(results[int(0.9 * simulations)], 2),
                    "pessimistic_10th_percentile": round(results[int(0.1 * simulations)], 2),
                    "most_likely_range": {
                        "lower": round(results[int(0.4 * simulations)], 2),
                        "upper": round(results[int(0.6 * simulations)], 2)
                    }
                },
                "llm_analysis": llm_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Monte Carlo analysis completed with {simulations} simulations")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Monte Carlo analysis failed: {e}")
            return {"error": str(e), "statistics": {}}
    
    async def stakeholder_influence_scoring(self, stakeholders: List[Dict]) -> Dict[str, Any]:
        """
        Calculate sophisticated stakeholder influence scores using mathematical models
        """
        try:
            if not stakeholders:
                return {"error": "No stakeholders provided", "scores": []}
            
            calculation_prompt = f"""
Analyze stakeholder influence using advanced scoring methodology:

Stakeholders: {json.dumps(stakeholders, indent=2)}

Calculate:
1. Multi-factor influence scores using weighted criteria
2. Network effect analysis
3. Decision-making power coefficients
4. Relationship strength indicators
5. Overall stakeholder portfolio risk assessment

Use mathematical models including:
- Weighted scoring algorithms
- Network centrality measures
- Risk-adjusted influence metrics
- Portfolio optimization theory

Provide mathematical justification for each scoring component.
"""
            
            llm_analysis = await self._execute_math_calculation(calculation_prompt)
            
            # Direct mathematical scoring
            scored_stakeholders = []
            total_influence = 0
            
            for stakeholder in stakeholders:
                # Multi-factor scoring algorithm
                influence_weights = {"high": 3, "medium": 2, "low": 1}
                relationship_weights = {"positive": 1.2, "neutral": 1.0, "negative": 0.7}
                
                base_influence = influence_weights.get(stakeholder.get('influence', 'medium').lower(), 2)
                relationship_multiplier = relationship_weights.get(stakeholder.get('relationship', 'neutral').lower(), 1.0)
                
                # Role-based authority scoring
                role = stakeholder.get('role', '').lower()
                authority_bonus = 0
                if any(exec_role in role for exec_role in ['ceo', 'cfo', 'cto', 'president', 'director']):
                    authority_bonus = 1.5
                elif any(mgr_role in role for mgr_role in ['manager', 'lead', 'head']):
                    authority_bonus = 1.2
                elif any(vp_role in role for vp_role in ['vp', 'vice president']):
                    authority_bonus = 1.8
                
                # Network effect calculation (simplified)
                network_factor = min(len(stakeholders) * 0.1, 1.5)  # Network effects
                
                final_score = (base_influence + authority_bonus) * relationship_multiplier * network_factor
                
                scored_stakeholder = {
                    "name": stakeholder.get('name', 'Unknown'),
                    "role": stakeholder.get('role', 'Unknown'),
                    "original_influence": stakeholder.get('influence', 'medium'),
                    "original_relationship": stakeholder.get('relationship', 'neutral'),
                    "calculated_score": round(final_score, 2),
                    "components": {
                        "base_influence": base_influence,
                        "authority_bonus": authority_bonus,
                        "relationship_multiplier": relationship_multiplier,
                        "network_factor": round(network_factor, 2)
                    }
                }
                
                scored_stakeholders.append(scored_stakeholder)
                total_influence += final_score
            
            # Sort by score
            scored_stakeholders.sort(key=lambda x: x['calculated_score'], reverse=True)
            
            # Portfolio analysis
            high_influence_count = len([s for s in scored_stakeholders if s['calculated_score'] >= 4.0])
            risk_concentration = high_influence_count / len(scored_stakeholders) if scored_stakeholders else 0
            
            analysis = {
                "stakeholder_scores": scored_stakeholders,
                "portfolio_metrics": {
                    "total_influence_score": round(total_influence, 2),
                    "average_influence": round(total_influence / len(scored_stakeholders), 2) if scored_stakeholders else 0,
                    "high_influence_stakeholders": high_influence_count,
                    "risk_concentration": round(risk_concentration, 2),
                    "stakeholder_diversity": len(set(s.get('role', '') for s in stakeholders))
                },
                "recommendations": self._generate_stakeholder_recommendations(scored_stakeholders, risk_concentration),
                "llm_analysis": llm_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Stakeholder influence analysis completed for {len(stakeholders)} stakeholders")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Stakeholder analysis failed: {e}")
            return {"error": str(e), "scores": []}
    
    async def growth_sustainability_analysis(self, current_revenue: float, target_growth: float, 
                                           market_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze growth rate sustainability using mathematical models
        """
        try:
            calculation_prompt = f"""
Analyze growth sustainability with these parameters:
- Current Revenue: ${current_revenue:,.2f}
- Target Growth Rate: {target_growth}%
- Market Size: {f'${market_size:,.2f}' if market_size else 'Not specified'}

Perform mathematical analysis:
1. Growth rate feasibility assessment
2. Market penetration implications
3. Compound annual growth rate (CAGR) projections
4. Resource requirement calculations
5. Risk-adjusted growth targets
6. Competitive dynamics modeling

Use mathematical frameworks including:
- Exponential growth models
- Market saturation curves
- Resource allocation optimization
- Probability-weighted scenarios

Provide quantitative analysis with supporting mathematics.
"""
            
            llm_analysis = await self._execute_math_calculation(calculation_prompt)
            
            # Mathematical sustainability analysis
            sustainability_score = 1.0
            factors = {}
            
            # Growth rate assessment
            if target_growth > 50:
                sustainability_score *= 0.3
                factors["growth_rate"] = "Extremely aggressive (>50%)"
            elif target_growth > 30:
                sustainability_score *= 0.6
                factors["growth_rate"] = "Very aggressive (30-50%)"
            elif target_growth > 15:
                sustainability_score *= 0.9
                factors["growth_rate"] = "Ambitious (15-30%)"
            else:
                sustainability_score *= 1.0
                factors["growth_rate"] = "Conservative (<15%)"
            
            # Market size considerations
            if market_size:
                market_share = current_revenue / market_size
                if market_share > 0.3:
                    sustainability_score *= 0.5
                    factors["market_position"] = "Dominant position (>30% share)"
                elif market_share > 0.1:
                    sustainability_score *= 0.8
                    factors["market_position"] = "Strong position (10-30% share)"
                else:
                    sustainability_score *= 1.0
                    factors["market_position"] = "Growth opportunity (<10% share)"
            
            # Calculate required compound growth
            projected_revenue_1yr = current_revenue * (1 + target_growth/100)
            projected_revenue_3yr = current_revenue * ((1 + target_growth/100) ** 3)
            projected_revenue_5yr = current_revenue * ((1 + target_growth/100) ** 5)
            
            analysis = {
                "input_parameters": {
                    "current_revenue": current_revenue,
                    "target_growth_rate": target_growth,
                    "market_size": market_size
                },
                "sustainability_assessment": {
                    "sustainability_score": round(sustainability_score, 2),
                    "assessment": "High" if sustainability_score >= 0.8 else "Medium" if sustainability_score >= 0.5 else "Low",
                    "contributing_factors": factors
                },
                "projections": {
                    "1_year": round(projected_revenue_1yr, 2),
                    "3_year": round(projected_revenue_3yr, 2),
                    "5_year": round(projected_revenue_5yr, 2),
                    "cagr_over_5_years": round(((projected_revenue_5yr / current_revenue) ** (1/5) - 1) * 100, 2)
                },
                "resource_implications": {
                    "revenue_multiple_1yr": round(projected_revenue_1yr / current_revenue, 2),
                    "revenue_multiple_3yr": round(projected_revenue_3yr / current_revenue, 2),
                    "required_efficiency_gains": max(0, round((target_growth - 10) / 2, 1)),  # Simplified model
                    "investment_requirement_estimate": round(current_revenue * (target_growth / 100) * 0.3, 2)  # 30% of growth
                },
                "recommendations": self._generate_growth_recommendations(sustainability_score, target_growth),
                "llm_analysis": llm_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Growth sustainability analysis completed (score: {sustainability_score:.2f})")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Growth sustainability analysis failed: {e}")
            return {"error": str(e), "assessment": {}}
    
    async def _execute_math_calculation(self, prompt: str) -> str:
        """Execute mathematical calculation using LLM"""
        try:
            if not self.pool_manager or not self.pool_manager.openai_client:
                return "LLM mathematical analysis not available (no OpenAI client)"
            
            response = await self.pool_manager.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert mathematical analyst and financial modeler. Provide detailed, accurate mathematical analysis with step-by-step calculations. Show your work and explain the mathematical reasoning behind each calculation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for mathematical precision
                max_tokens=1500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"❌ LLM math calculation failed: {e}")
            return f"Mathematical analysis failed: {e}"
    
    def _generate_stakeholder_recommendations(self, scored_stakeholders: List[Dict], risk_concentration: float) -> List[str]:
        """Generate stakeholder management recommendations"""
        recommendations = []
        
        if risk_concentration > 0.7:
            recommendations.append("High stakeholder risk concentration - diversify engagement strategy")
        
        if scored_stakeholders:
            top_stakeholder = scored_stakeholders[0]
            recommendations.append(f"Prioritize engagement with {top_stakeholder['name']} (highest influence score: {top_stakeholder['calculated_score']})")
        
        high_negative = [s for s in scored_stakeholders if s.get('original_relationship', '').lower() == 'negative' and s['calculated_score'] >= 3.0]
        if high_negative:
            recommendations.append(f"Address {len(high_negative)} high-influence negative relationships immediately")
        
        return recommendations
    
    def _generate_growth_recommendations(self, sustainability_score: float, target_growth: float) -> List[str]:
        """Generate growth strategy recommendations"""
        recommendations = []
        
        if sustainability_score < 0.5:
            recommendations.append("Consider reducing growth targets or extending timeline for better sustainability")
        
        if target_growth > 30:
            recommendations.append("Extremely high growth target requires exceptional execution and market conditions")
        
        if sustainability_score >= 0.8:
            recommendations.append("Growth target appears sustainable with proper resource allocation")
        
        recommendations.append("Monitor competitive dynamics and market saturation indicators")
        recommendations.append("Implement milestone-based growth tracking with contingency planning")
        
        return recommendations

# Global instance
llm_math_engine = None

def get_llm_math_engine(pool_manager=None):
    """Get or create LLM math engine instance"""
    global llm_math_engine
    if llm_math_engine is None:
        llm_math_engine = LLMMathEngine(pool_manager)
    return llm_math_engine














































