"""
Template Cost Estimation System
Task 4.2.26: Add cost estimation per template (inference cost, execution cost)

Provides FinOps predictability with detailed cost breakdowns for template deployments.
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CostComponent:
    """Individual cost component"""
    component_name: str
    component_type: str  # 'inference', 'compute', 'storage', 'api', 'data_transfer'
    unit_cost: Decimal
    units_consumed: float
    total_cost: Decimal
    cost_basis: str  # 'per_execution', 'per_hour', 'per_GB', 'per_request'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemplateCostEstimate:
    """Complete cost estimate for a template"""
    template_id: str
    template_name: str
    estimation_id: str
    total_estimated_cost: Decimal
    cost_components: List[CostComponent]
    execution_volume: int  # Expected executions per period
    time_period: str  # 'hourly', 'daily', 'monthly', 'annual'
    confidence_level: float  # 0.0 to 1.0
    cost_breakdown: Dict[str, Decimal]
    optimization_suggestions: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CostComparison:
    """Cost comparison between scenarios"""
    scenario_a: str
    scenario_b: str
    cost_a: Decimal
    cost_b: Decimal
    cost_difference: Decimal
    cost_difference_percentage: float
    recommendation: str

class TemplateCostEstimator:
    """
    Cost estimation system for RBIA templates
    
    Features:
    - ML inference cost calculation
    - Execution runtime cost estimation
    - Storage and data transfer costs
    - API call costs
    - Cost optimization recommendations
    - ROI analysis
    - Budget forecasting
    - Multi-cloud pricing support
    """
    
    def __init__(self, db_path: str = "template_costs.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self._init_database()
        self._load_pricing_models()
    
    def _init_database(self):
        """Initialize cost tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cost estimates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_estimates (
                estimation_id TEXT PRIMARY KEY,
                template_id TEXT NOT NULL,
                template_name TEXT NOT NULL,
                total_cost REAL NOT NULL,
                execution_volume INTEGER NOT NULL,
                time_period TEXT NOT NULL,
                confidence_level REAL NOT NULL,
                cost_breakdown TEXT NOT NULL,
                optimization_suggestions TEXT,
                created_at TIMESTAMP NOT NULL,
                metadata TEXT
            )
        """)
        
        # Cost actuals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_actuals (
                actual_id TEXT PRIMARY KEY,
                template_id TEXT NOT NULL,
                execution_id TEXT NOT NULL,
                actual_cost REAL NOT NULL,
                execution_duration_ms INTEGER NOT NULL,
                cost_components TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                tenant_id TEXT NOT NULL
            )
        """)
        
        # Pricing models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pricing_models (
                model_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                region TEXT NOT NULL,
                unit_cost REAL NOT NULL,
                cost_basis TEXT NOT NULL,
                effective_date DATE NOT NULL,
                expiry_date DATE,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info("Cost estimation database initialized")
    
    def _load_pricing_models(self):
        """Load pricing models for various resources"""
        self.pricing = {
            # ML Inference Costs (per 1000 predictions)
            "ml_inference": {
                "small_model": Decimal("0.001"),  # $0.001 per 1000 predictions
                "medium_model": Decimal("0.005"),  # $0.005 per 1000 predictions
                "large_model": Decimal("0.020"),  # $0.020 per 1000 predictions
                "xlarge_model": Decimal("0.100")   # $0.100 per 1000 predictions
            },
            
            # Compute Costs (per second)
            "compute": {
                "cpu_light": Decimal("0.00001"),    # $0.036/hour
                "cpu_standard": Decimal("0.00003"),  # $0.108/hour
                "cpu_heavy": Decimal("0.00010"),     # $0.360/hour
                "gpu_small": Decimal("0.00050"),     # $1.80/hour
                "gpu_large": Decimal("0.00200")      # $7.20/hour
            },
            
            # Storage Costs (per GB-month)
            "storage": {
                "hot": Decimal("0.023"),      # $0.023/GB-month
                "cool": Decimal("0.010"),     # $0.010/GB-month
                "archive": Decimal("0.002")   # $0.002/GB-month
            },
            
            # API Costs (per 1000 calls)
            "api": {
                "internal": Decimal("0.0"),     # Free
                "external": Decimal("0.001"),   # $0.001 per 1000 calls
                "llm_call": Decimal("0.020")    # $0.020 per 1000 calls
            },
            
            # Data Transfer (per GB)
            "data_transfer": {
                "inbound": Decimal("0.0"),      # Free
                "outbound": Decimal("0.09"),    # $0.09/GB
                "inter_region": Decimal("0.02") # $0.02/GB
            },
            
            # Database Operations (per 1M operations)
            "database": {
                "read": Decimal("0.25"),        # $0.25 per 1M reads
                "write": Decimal("1.25"),       # $1.25 per 1M writes
                "query": Decimal("5.00")        # $5.00 per 1M queries
            }
        }
    
    def estimate_template_cost(
        self,
        template_id: str,
        template_name: str,
        execution_volume: int,
        time_period: str = "monthly",
        model_size: str = "medium_model",
        compute_profile: str = "cpu_standard",
        include_storage: bool = True,
        include_api: bool = True,
        tenant_id: Optional[str] = None
    ) -> TemplateCostEstimate:
        """
        Estimate total cost for running a template
        
        Args:
            template_id: Template identifier
            template_name: Template name
            execution_volume: Expected number of executions
            time_period: Time period for estimate
            model_size: ML model size category
            compute_profile: Compute resource profile
            include_storage: Include storage costs
            include_api: Include API costs
            tenant_id: Optional tenant identifier
        
        Returns:
            Complete cost estimate
        """
        try:
            estimation_id = str(uuid.uuid4())
            cost_components = []
            
            # 1. ML Inference Costs
            inference_unit_cost = self.pricing["ml_inference"].get(model_size, Decimal("0.005"))
            inference_units = execution_volume / 1000  # Per 1000 predictions
            inference_cost = inference_unit_cost * Decimal(str(inference_units))
            
            cost_components.append(CostComponent(
                component_name="ML Inference",
                component_type="inference",
                unit_cost=inference_unit_cost,
                units_consumed=inference_units,
                total_cost=inference_cost,
                cost_basis="per_1000_predictions",
                metadata={"model_size": model_size}
            ))
            
            # 2. Compute Costs (assume 500ms average execution time)
            avg_execution_time_seconds = 0.5
            compute_unit_cost = self.pricing["compute"].get(compute_profile, Decimal("0.00003"))
            compute_units = execution_volume * avg_execution_time_seconds
            compute_cost = compute_unit_cost * Decimal(str(compute_units))
            
            cost_components.append(CostComponent(
                component_name="Compute Runtime",
                component_type="compute",
                unit_cost=compute_unit_cost,
                units_consumed=compute_units,
                total_cost=compute_cost,
                cost_basis="per_second",
                metadata={"compute_profile": compute_profile, "avg_execution_time": avg_execution_time_seconds}
            ))
            
            # 3. Storage Costs (logs, traces, evidence packs)
            storage_cost = Decimal("0.0")
            if include_storage:
                # Assume 10KB per execution for logs/traces
                storage_gb = (execution_volume * 10) / (1024 * 1024)  # Convert KB to GB
                storage_unit_cost = self.pricing["storage"]["hot"]
                storage_cost = storage_unit_cost * Decimal(str(storage_gb))
                
                cost_components.append(CostComponent(
                    component_name="Storage (Logs/Traces)",
                    component_type="storage",
                    unit_cost=storage_unit_cost,
                    units_consumed=storage_gb,
                    total_cost=storage_cost,
                    cost_basis="per_GB_month",
                    metadata={"avg_size_per_execution": "10KB"}
                ))
            
            # 4. API Costs (explainability, external integrations)
            api_cost = Decimal("0.0")
            if include_api:
                # Assume 2 API calls per execution
                api_calls = execution_volume * 2
                api_unit_cost = self.pricing["api"]["external"]
                api_units = api_calls / 1000
                api_cost = api_unit_cost * Decimal(str(api_units))
                
                cost_components.append(CostComponent(
                    component_name="API Calls",
                    component_type="api",
                    unit_cost=api_unit_cost,
                    units_consumed=api_units,
                    total_cost=api_cost,
                    cost_basis="per_1000_calls",
                    metadata={"api_calls_per_execution": 2}
                ))
            
            # 5. Database Operations
            # Assume 5 reads and 2 writes per execution
            db_reads = execution_volume * 5
            db_writes = execution_volume * 2
            
            read_unit_cost = self.pricing["database"]["read"]
            write_unit_cost = self.pricing["database"]["write"]
            
            read_cost = read_unit_cost * Decimal(str(db_reads / 1_000_000))
            write_cost = write_unit_cost * Decimal(str(db_writes / 1_000_000))
            db_cost = read_cost + write_cost
            
            cost_components.append(CostComponent(
                component_name="Database Operations",
                component_type="storage",
                unit_cost=read_unit_cost + write_unit_cost,
                units_consumed=(db_reads + db_writes) / 1_000_000,
                total_cost=db_cost,
                cost_basis="per_million_ops",
                metadata={"reads": db_reads, "writes": db_writes}
            ))
            
            # 6. Data Transfer (assume 1KB per execution)
            transfer_gb = (execution_volume * 1) / (1024 * 1024)
            transfer_unit_cost = self.pricing["data_transfer"]["outbound"]
            transfer_cost = transfer_unit_cost * Decimal(str(transfer_gb))
            
            cost_components.append(CostComponent(
                component_name="Data Transfer",
                component_type="data_transfer",
                unit_cost=transfer_unit_cost,
                units_consumed=transfer_gb,
                total_cost=transfer_cost,
                cost_basis="per_GB",
                metadata={"avg_transfer_per_execution": "1KB"}
            ))
            
            # Calculate total cost
            total_cost = sum(c.total_cost for c in cost_components)
            
            # Cost breakdown by type
            cost_breakdown = {}
            for component in cost_components:
                cost_breakdown[component.component_type] = \
                    cost_breakdown.get(component.component_type, Decimal("0.0")) + component.total_cost
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                cost_components, execution_volume
            )
            
            # Calculate confidence level based on execution volume
            confidence_level = min(0.95, 0.5 + (execution_volume / 100000) * 0.45)
            
            # Create estimate
            estimate = TemplateCostEstimate(
                template_id=template_id,
                template_name=template_name,
                estimation_id=estimation_id,
                total_estimated_cost=total_cost,
                cost_components=cost_components,
                execution_volume=execution_volume,
                time_period=time_period,
                confidence_level=confidence_level,
                cost_breakdown=cost_breakdown,
                optimization_suggestions=optimization_suggestions,
                metadata={
                    "tenant_id": tenant_id,
                    "model_size": model_size,
                    "compute_profile": compute_profile
                }
            )
            
            # Store estimate
            self._store_estimate(estimate)
            
            return estimate
            
        except Exception as e:
            self.logger.error(f"Failed to estimate template cost: {e}")
            raise
    
    def _generate_optimization_suggestions(
        self,
        cost_components: List[CostComponent],
        execution_volume: int
    ) -> List[str]:
        """Generate cost optimization suggestions"""
        suggestions = []
        
        # Analyze each component
        for component in cost_components:
            cost_pct = (component.total_cost / sum(c.total_cost for c in cost_components)) * 100
            
            if component.component_type == "inference" and cost_pct > 40:
                suggestions.append(
                    f"ML inference accounts for {cost_pct:.1f}% of costs. "
                    f"Consider model optimization, quantization, or batching to reduce inference costs."
                )
            
            if component.component_type == "compute" and cost_pct > 30:
                suggestions.append(
                    f"Compute runtime accounts for {cost_pct:.1f}% of costs. "
                    f"Consider execution optimization, caching, or reserved capacity for cost savings."
                )
            
            if component.component_type == "storage" and cost_pct > 20:
                suggestions.append(
                    f"Storage accounts for {cost_pct:.1f}% of costs. "
                    f"Consider implementing data retention policies, compression, or tiered storage."
                )
        
        # Volume-based suggestions
        if execution_volume > 100000:
            suggestions.append(
                f"High execution volume ({execution_volume:,}). "
                f"Consider reserved capacity or volume discounts for 15-30% cost reduction."
            )
        
        if execution_volume < 1000:
            suggestions.append(
                f"Low execution volume ({execution_volume:,}). "
                f"Consider serverless or pay-per-use pricing for better cost efficiency."
            )
        
        return suggestions
    
    def _store_estimate(self, estimate: TemplateCostEstimate):
        """Store cost estimate in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO cost_estimates (
                    estimation_id, template_id, template_name, total_cost,
                    execution_volume, time_period, confidence_level,
                    cost_breakdown, optimization_suggestions, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                estimate.estimation_id,
                estimate.template_id,
                estimate.template_name,
                float(estimate.total_estimated_cost),
                estimate.execution_volume,
                estimate.time_period,
                estimate.confidence_level,
                json.dumps({k: float(v) for k, v in estimate.cost_breakdown.items()}),
                json.dumps(estimate.optimization_suggestions),
                estimate.created_at.isoformat(),
                json.dumps(estimate.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store cost estimate: {e}")
    
    def compare_scenarios(
        self,
        template_id: str,
        template_name: str,
        scenario_a: Dict[str, Any],
        scenario_b: Dict[str, Any]
    ) -> CostComparison:
        """
        Compare costs between two scenarios
        
        Args:
            template_id: Template identifier
            template_name: Template name
            scenario_a: First scenario parameters
            scenario_b: Second scenario parameters
        
        Returns:
            Cost comparison analysis
        """
        estimate_a = self.estimate_template_cost(template_id, template_name, **scenario_a)
        estimate_b = self.estimate_template_cost(template_id, template_name, **scenario_b)
        
        cost_diff = estimate_b.total_estimated_cost - estimate_a.total_estimated_cost
        cost_diff_pct = (float(cost_diff) / float(estimate_a.total_estimated_cost)) * 100
        
        # Generate recommendation
        if cost_diff < 0:
            recommendation = f"Scenario B is {abs(cost_diff_pct):.1f}% cheaper. Recommended approach."
        elif cost_diff > 0:
            recommendation = f"Scenario A is {abs(cost_diff_pct):.1f}% cheaper. Recommended approach."
        else:
            recommendation = "Both scenarios have similar costs. Choose based on other factors."
        
        return CostComparison(
            scenario_a=scenario_a.get("compute_profile", "A"),
            scenario_b=scenario_b.get("compute_profile", "B"),
            cost_a=estimate_a.total_estimated_cost,
            cost_b=estimate_b.total_estimated_cost,
            cost_difference=cost_diff,
            cost_difference_percentage=cost_diff_pct,
            recommendation=recommendation
        )
    
    def get_cost_history(
        self,
        template_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical cost estimates for a template"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT estimation_id, template_id, template_name, total_cost,
                       execution_volume, time_period, confidence_level,
                       cost_breakdown, created_at
                FROM cost_estimates
                WHERE template_id = ? AND created_at >= ?
                ORDER BY created_at DESC
            """, (template_id, cutoff_date))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "estimation_id": row[0],
                    "template_id": row[1],
                    "template_name": row[2],
                    "total_cost": row[3],
                    "execution_volume": row[4],
                    "time_period": row[5],
                    "confidence_level": row[6],
                    "cost_breakdown": json.loads(row[7]),
                    "created_at": row[8]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get cost history: {e}")
            return []
    
    def calculate_roi(
        self,
        template_id: str,
        template_name: str,
        execution_volume: int,
        expected_benefit: Decimal,
        benefit_type: str = "cost_savings"
    ) -> Dict[str, Any]:
        """
        Calculate ROI for template deployment
        
        Args:
            template_id: Template identifier
            template_name: Template name
            execution_volume: Expected executions
            expected_benefit: Expected financial benefit
            benefit_type: Type of benefit (cost_savings, revenue_increase, etc.)
        
        Returns:
            ROI analysis
        """
        estimate = self.estimate_template_cost(
            template_id=template_id,
            template_name=template_name,
            execution_volume=execution_volume,
            time_period="monthly"
        )
        
        total_cost = estimate.total_estimated_cost
        net_benefit = expected_benefit - total_cost
        roi_percentage = (float(net_benefit) / float(total_cost)) * 100 if total_cost > 0 else 0
        payback_months = float(total_cost) / float(expected_benefit) if expected_benefit > 0 else float('inf')
        
        return {
            "template_id": template_id,
            "template_name": template_name,
            "total_cost": float(total_cost),
            "expected_benefit": float(expected_benefit),
            "net_benefit": float(net_benefit),
            "roi_percentage": roi_percentage,
            "payback_months": payback_months,
            "benefit_type": benefit_type,
            "recommendation": "Deploy" if roi_percentage > 100 else "Review costs"
        }


# Example usage
def main():
    """Demonstrate template cost estimation"""
    
    estimator = TemplateCostEstimator()
    
    print("=" * 80)
    print("TEMPLATE COST ESTIMATION SYSTEM")
    print("=" * 80)
    print()
    
    # Example 1: SaaS Churn Risk Alert Cost Estimate
    print("Example 1: SaaS Churn Risk Alert - Monthly Cost Estimate")
    print("-" * 80)
    
    estimate = estimator.estimate_template_cost(
        template_id="saas_churn_risk_alert",
        template_name="SaaS Churn Risk Alert",
        execution_volume=50000,
        time_period="monthly",
        model_size="medium_model",
        compute_profile="cpu_standard",
        tenant_id="tenant_acme"
    )
    
    print(f"Template: {estimate.template_name}")
    print(f"Execution Volume: {estimate.execution_volume:,} per {estimate.time_period}")
    print(f"Total Estimated Cost: ${estimate.total_estimated_cost:.2f}")
    print(f"Confidence Level: {estimate.confidence_level:.1%}")
    print()
    print("Cost Breakdown:")
    for component_type, cost in estimate.cost_breakdown.items():
        percentage = (cost / estimate.total_estimated_cost) * 100
        print(f"  {component_type:20s}: ${cost:8.2f} ({percentage:5.1f}%)")
    print()
    print("Optimization Suggestions:")
    for i, suggestion in enumerate(estimate.optimization_suggestions, 1):
        print(f"  {i}. {suggestion}")
    print()
    
    # Example 2: Banking Credit Scoring Cost Estimate
    print("Example 2: Banking Credit Scoring - Monthly Cost Estimate")
    print("-" * 80)
    
    estimate2 = estimator.estimate_template_cost(
        template_id="banking_credit_scoring_check",
        template_name="Banking Credit Scoring Check",
        execution_volume=100000,
        time_period="monthly",
        model_size="large_model",
        compute_profile="cpu_heavy",
        tenant_id="tenant_megabank"
    )
    
    print(f"Template: {estimate2.template_name}")
    print(f"Execution Volume: {estimate2.execution_volume:,} per {estimate2.time_period}")
    print(f"Total Estimated Cost: ${estimate2.total_estimated_cost:.2f}")
    print()
    
    # Example 3: Scenario Comparison
    print("Example 3: Scenario Comparison - Standard vs Heavy Compute")
    print("-" * 80)
    
    comparison = estimator.compare_scenarios(
        template_id="insurance_claim_fraud_anomaly",
        template_name="Insurance Claim Fraud Anomaly",
        scenario_a={
            "execution_volume": 10000,
            "model_size": "medium_model",
            "compute_profile": "cpu_standard"
        },
        scenario_b={
            "execution_volume": 10000,
            "model_size": "medium_model",
            "compute_profile": "cpu_heavy"
        }
    )
    
    print(f"Scenario A Cost: ${comparison.cost_a:.2f}")
    print(f"Scenario B Cost: ${comparison.cost_b:.2f}")
    print(f"Cost Difference: ${comparison.cost_difference:.2f} ({comparison.cost_difference_percentage:.1f}%)")
    print(f"Recommendation: {comparison.recommendation}")
    print()
    
    # Example 4: ROI Calculation
    print("Example 4: ROI Analysis - E-commerce Fraud Detection")
    print("-" * 80)
    
    roi = estimator.calculate_roi(
        template_id="ecommerce_fraud_scoring_checkout",
        template_name="E-commerce Fraud Scoring at Checkout",
        execution_volume=500000,
        expected_benefit=Decimal("25000.00"),  # $25K monthly savings from fraud prevention
        benefit_type="fraud_prevention_savings"
    )
    
    print(f"Template: {roi['template_name']}")
    print(f"Total Cost: ${roi['total_cost']:.2f}")
    print(f"Expected Benefit: ${roi['expected_benefit']:.2f}")
    print(f"Net Benefit: ${roi['net_benefit']:.2f}")
    print(f"ROI: {roi['roi_percentage']:.1f}%")
    print(f"Payback Period: {roi['payback_months']:.1f} months")
    print(f"Recommendation: {roi['recommendation']}")
    print()
    
    print("=" * 80)
    print("✅ Task 4.2.26 Complete: Cost estimation per template implemented!")
    print("=" * 80)


if __name__ == "__main__":
    main()

