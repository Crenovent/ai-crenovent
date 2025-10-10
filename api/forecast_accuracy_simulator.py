"""
Task 4.4.22: Build "what-if" simulator for forecast accuracy with/without RBIA
- Quantify RBIA impact
- Simulation engine for CFO persona
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import random
import statistics

app = FastAPI(title="RBIA Forecast Accuracy What-If Simulator")
logger = logging.getLogger(__name__)

class SimulationScenario(str, Enum):
    WITHOUT_RBIA = "without_rbia"
    WITH_RBIA = "with_rbia"
    RBIA_IMPROVEMENT = "rbia_improvement"

class ForecastDomain(str, Enum):
    REVENUE = "revenue"
    PIPELINE = "pipeline"
    BOOKINGS = "bookings"
    CHURN = "churn"

class SimulationInput(BaseModel):
    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Simulation parameters
    forecast_domain: ForecastDomain
    historical_data: List[float]  # Actual historical values
    forecast_periods: int = 12    # Number of periods to simulate
    
    # RBIA improvement factors
    accuracy_improvement_pct: float = 15.0  # % improvement with RBIA
    variance_reduction_pct: float = 25.0    # % variance reduction
    bias_correction_factor: float = 0.1     # Bias correction
    
    # Simulation settings
    monte_carlo_runs: int = 1000
    confidence_level: float = 0.95
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SimulationResult(BaseModel):
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    simulation_id: str
    tenant_id: str
    
    # Scenario results
    without_rbia_forecast: List[float]
    with_rbia_forecast: List[float]
    
    # Accuracy metrics
    without_rbia_mae: float  # Mean Absolute Error
    with_rbia_mae: float
    accuracy_improvement: float
    
    # Variance metrics
    without_rbia_variance: float
    with_rbia_variance: float
    variance_reduction: float
    
    # Confidence intervals
    without_rbia_ci_lower: List[float]
    without_rbia_ci_upper: List[float]
    with_rbia_ci_lower: List[float]
    with_rbia_ci_upper: List[float]
    
    # Business impact
    forecast_value_improvement: float
    risk_reduction_score: float
    
    # Metadata
    simulation_date: datetime = Field(default_factory=datetime.utcnow)
    monte_carlo_runs: int
    confidence_level: float

class ComparisonReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    
    # Summary metrics
    total_simulations: int
    average_accuracy_improvement: float
    average_variance_reduction: float
    average_value_improvement: float
    
    # ROI calculation
    estimated_annual_value: float
    rbia_implementation_cost: float = 100000.0  # Default estimate
    roi_percentage: float
    payback_months: float
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
simulation_inputs_store: Dict[str, SimulationInput] = {}
simulation_results_store: Dict[str, SimulationResult] = {}
comparison_reports_store: Dict[str, ComparisonReport] = {}

class ForecastAccuracySimulator:
    def __init__(self):
        self.random_seed = 42
        
    def generate_baseline_forecast(self, historical_data: List[float], 
                                 periods: int) -> List[float]:
        """Generate baseline forecast without RBIA"""
        if not historical_data:
            return [0.0] * periods
            
        # Simple trend-based forecast with noise
        trend = (historical_data[-1] - historical_data[0]) / len(historical_data)
        base_value = historical_data[-1]
        
        forecast = []
        for i in range(periods):
            # Add trend and random noise
            noise_factor = random.uniform(-0.1, 0.1)
            value = base_value + (trend * (i + 1)) * (1 + noise_factor)
            forecast.append(max(0, value))  # Ensure non-negative
            
        return forecast
    
    def apply_rbia_improvements(self, baseline_forecast: List[float],
                              accuracy_improvement: float,
                              variance_reduction: float,
                              bias_correction: float) -> List[float]:
        """Apply RBIA improvements to baseline forecast"""
        improved_forecast = []
        
        for value in baseline_forecast:
            # Apply accuracy improvement (reduce error)
            improved_value = value * (1 + accuracy_improvement / 100)
            
            # Apply bias correction
            improved_value = improved_value * (1 - bias_correction)
            
            # Reduce variance (make predictions more stable)
            variance_factor = 1 - (variance_reduction / 100)
            noise = random.uniform(-0.05, 0.05) * variance_factor
            improved_value = improved_value * (1 + noise)
            
            improved_forecast.append(max(0, improved_value))
            
        return improved_forecast
    
    def calculate_mae(self, predicted: List[float], actual: List[float]) -> float:
        """Calculate Mean Absolute Error"""
        if len(predicted) != len(actual) or not predicted:
            return 0.0
            
        errors = [abs(p - a) for p, a in zip(predicted, actual)]
        return sum(errors) / len(errors)
    
    def monte_carlo_simulation(self, simulation_input: SimulationInput) -> SimulationResult:
        """Run Monte Carlo simulation for forecast accuracy"""
        random.seed(self.random_seed)
        
        # Store all simulation runs
        without_rbia_runs = []
        with_rbia_runs = []
        
        for _ in range(simulation_input.monte_carlo_runs):
            # Generate baseline forecast
            baseline = self.generate_baseline_forecast(
                simulation_input.historical_data,
                simulation_input.forecast_periods
            )
            
            # Apply RBIA improvements
            rbia_improved = self.apply_rbia_improvements(
                baseline,
                simulation_input.accuracy_improvement_pct,
                simulation_input.variance_reduction_pct,
                simulation_input.bias_correction_factor
            )
            
            without_rbia_runs.append(baseline)
            with_rbia_runs.append(rbia_improved)
        
        # Calculate mean forecasts
        without_rbia_forecast = [
            statistics.mean([run[i] for run in without_rbia_runs])
            for i in range(simulation_input.forecast_periods)
        ]
        
        with_rbia_forecast = [
            statistics.mean([run[i] for run in with_rbia_runs])
            for i in range(simulation_input.forecast_periods)
        ]
        
        # Calculate confidence intervals
        confidence = simulation_input.confidence_level
        alpha = (1 - confidence) / 2
        
        without_rbia_ci_lower = []
        without_rbia_ci_upper = []
        with_rbia_ci_lower = []
        with_rbia_ci_upper = []
        
        for i in range(simulation_input.forecast_periods):
            without_values = sorted([run[i] for run in without_rbia_runs])
            with_values = sorted([run[i] for run in with_rbia_runs])
            
            lower_idx = int(alpha * len(without_values))
            upper_idx = int((1 - alpha) * len(without_values))
            
            without_rbia_ci_lower.append(without_values[lower_idx])
            without_rbia_ci_upper.append(without_values[upper_idx])
            with_rbia_ci_lower.append(with_values[lower_idx])
            with_rbia_ci_upper.append(with_values[upper_idx])
        
        # Calculate metrics (using synthetic "actual" values for demonstration)
        synthetic_actual = [
            val * random.uniform(0.95, 1.05) 
            for val in simulation_input.historical_data[-simulation_input.forecast_periods:]
        ] if len(simulation_input.historical_data) >= simulation_input.forecast_periods else without_rbia_forecast
        
        without_rbia_mae = self.calculate_mae(without_rbia_forecast, synthetic_actual)
        with_rbia_mae = self.calculate_mae(with_rbia_forecast, synthetic_actual)
        accuracy_improvement = ((without_rbia_mae - with_rbia_mae) / without_rbia_mae * 100) if without_rbia_mae > 0 else 0
        
        # Calculate variances
        without_rbia_variance = statistics.variance(without_rbia_forecast) if len(without_rbia_forecast) > 1 else 0
        with_rbia_variance = statistics.variance(with_rbia_forecast) if len(with_rbia_forecast) > 1 else 0
        variance_reduction = ((without_rbia_variance - with_rbia_variance) / without_rbia_variance * 100) if without_rbia_variance > 0 else 0
        
        # Calculate business impact
        forecast_value_improvement = sum(with_rbia_forecast) - sum(without_rbia_forecast)
        risk_reduction_score = min(100, variance_reduction + accuracy_improvement)
        
        return SimulationResult(
            simulation_id=simulation_input.simulation_id,
            tenant_id=simulation_input.tenant_id,
            without_rbia_forecast=without_rbia_forecast,
            with_rbia_forecast=with_rbia_forecast,
            without_rbia_mae=without_rbia_mae,
            with_rbia_mae=with_rbia_mae,
            accuracy_improvement=accuracy_improvement,
            without_rbia_variance=without_rbia_variance,
            with_rbia_variance=with_rbia_variance,
            variance_reduction=variance_reduction,
            without_rbia_ci_lower=without_rbia_ci_lower,
            without_rbia_ci_upper=without_rbia_ci_upper,
            with_rbia_ci_lower=with_rbia_ci_lower,
            with_rbia_ci_upper=with_rbia_ci_upper,
            forecast_value_improvement=forecast_value_improvement,
            risk_reduction_score=risk_reduction_score,
            monte_carlo_runs=simulation_input.monte_carlo_runs,
            confidence_level=simulation_input.confidence_level
        )

# Global simulator instance
simulator = ForecastAccuracySimulator()

@app.post("/forecast-simulator/run", response_model=SimulationResult)
async def run_forecast_simulation(simulation_input: SimulationInput):
    """Run forecast accuracy what-if simulation"""
    
    if len(simulation_input.historical_data) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 historical data points")
    
    if simulation_input.forecast_periods < 1:
        raise HTTPException(status_code=400, detail="Forecast periods must be at least 1")
    
    # Store input
    simulation_inputs_store[simulation_input.simulation_id] = simulation_input
    
    # Run simulation
    result = simulator.monte_carlo_simulation(simulation_input)
    
    # Store result
    simulation_results_store[result.result_id] = result
    
    logger.info(f"✅ Completed forecast simulation {simulation_input.simulation_id} - Accuracy improvement: {result.accuracy_improvement:.2f}%")
    return result

@app.get("/forecast-simulator/results/{simulation_id}")
async def get_simulation_result(simulation_id: str):
    """Get simulation result by ID"""
    
    result = next(
        (r for r in simulation_results_store.values() if r.simulation_id == simulation_id),
        None
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Simulation result not found")
    
    return result

@app.get("/forecast-simulator/tenant/{tenant_id}/results")
async def get_tenant_simulations(tenant_id: str):
    """Get all simulation results for a tenant"""
    
    tenant_results = [
        result for result in simulation_results_store.values()
        if result.tenant_id == tenant_id
    ]
    
    return {
        "tenant_id": tenant_id,
        "simulation_count": len(tenant_results),
        "results": tenant_results
    }

@app.post("/forecast-simulator/comparison-report", response_model=ComparisonReport)
async def generate_comparison_report(tenant_id: str):
    """Generate comparison report for tenant simulations"""
    
    tenant_results = [
        result for result in simulation_results_store.values()
        if result.tenant_id == tenant_id
    ]
    
    if not tenant_results:
        raise HTTPException(status_code=404, detail="No simulation results found for tenant")
    
    # Calculate averages
    avg_accuracy_improvement = statistics.mean([r.accuracy_improvement for r in tenant_results])
    avg_variance_reduction = statistics.mean([r.variance_reduction for r in tenant_results])
    avg_value_improvement = statistics.mean([r.forecast_value_improvement for r in tenant_results])
    
    # Calculate ROI
    estimated_annual_value = avg_value_improvement * 12  # Annualize monthly improvement
    implementation_cost = 100000.0  # Default RBIA implementation cost
    roi_percentage = ((estimated_annual_value - implementation_cost) / implementation_cost * 100) if implementation_cost > 0 else 0
    payback_months = (implementation_cost / (avg_value_improvement * 12 / 12)) if avg_value_improvement > 0 else float('inf')
    
    # Generate recommendations
    recommendations = []
    if avg_accuracy_improvement > 10:
        recommendations.append("High accuracy improvement potential - recommend RBIA implementation")
    if avg_variance_reduction > 15:
        recommendations.append("Significant risk reduction achievable with RBIA")
    if roi_percentage > 100:
        recommendations.append(f"Strong ROI of {roi_percentage:.1f}% - prioritize implementation")
    if payback_months < 12:
        recommendations.append(f"Fast payback period of {payback_months:.1f} months")
    
    report = ComparisonReport(
        tenant_id=tenant_id,
        total_simulations=len(tenant_results),
        average_accuracy_improvement=avg_accuracy_improvement,
        average_variance_reduction=avg_variance_reduction,
        average_value_improvement=avg_value_improvement,
        estimated_annual_value=estimated_annual_value,
        rbia_implementation_cost=implementation_cost,
        roi_percentage=roi_percentage,
        payback_months=payback_months,
        recommendations=recommendations
    )
    
    comparison_reports_store[report.report_id] = report
    
    logger.info(f"✅ Generated comparison report for tenant {tenant_id} - ROI: {roi_percentage:.1f}%")
    return report

@app.get("/forecast-simulator/summary")
async def get_simulation_summary():
    """Get summary of all simulations"""
    
    total_simulations = len(simulation_results_store)
    
    if total_simulations == 0:
        return {
            "total_simulations": 0,
            "average_improvements": {},
            "total_reports": 0
        }
    
    # Calculate overall averages
    all_results = list(simulation_results_store.values())
    avg_accuracy = statistics.mean([r.accuracy_improvement for r in all_results])
    avg_variance = statistics.mean([r.variance_reduction for r in all_results])
    avg_value = statistics.mean([r.forecast_value_improvement for r in all_results])
    
    return {
        "total_simulations": total_simulations,
        "average_improvements": {
            "accuracy_improvement": round(avg_accuracy, 2),
            "variance_reduction": round(avg_variance, 2),
            "value_improvement": round(avg_value, 2)
        },
        "total_reports": len(comparison_reports_store)
    }
