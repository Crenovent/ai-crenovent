"""
Task 4.3.40: Drift What-If Simulator
Service for simulating stress conditions and drift scenarios
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json
import numpy as np
import asyncio
from dataclasses import dataclass

# Import existing drift monitor
from dsl.operators.drift_bias_monitor import DriftBiasMonitor, DriftAlert

app = FastAPI(title="RBIA Drift What-If Simulator")
logger = logging.getLogger(__name__)

class StressScenario(str, Enum):
    DATA_SHIFT = "data_shift"
    CONCEPT_DRIFT = "concept_drift"
    COVARIATE_SHIFT = "covariate_shift"
    LABEL_SHIFT = "label_shift"
    FEATURE_CORRUPTION = "feature_corruption"
    SEASONAL_CHANGE = "seasonal_change"
    POPULATION_SHIFT = "population_shift"
    ADVERSARIAL_ATTACK = "adversarial_attack"

class SimulationSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"

class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class DriftSimulationRequest(BaseModel):
    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    model_id: str
    
    # Scenario configuration
    scenario: StressScenario
    severity: SimulationSeverity
    duration_hours: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week
    
    # Data configuration
    baseline_data: Dict[str, List[float]] = Field(description="Baseline feature data")
    sample_size: int = Field(default=1000, ge=100, le=10000)
    
    # Simulation parameters
    drift_intensity: float = Field(default=0.1, ge=0.01, le=1.0)
    noise_level: float = Field(default=0.05, ge=0.0, le=0.5)
    
    # What-if conditions
    conditions: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    description: str = ""
    created_by: str
    tags: List[str] = Field(default_factory=list)

class DriftSimulationResult(BaseModel):
    simulation_id: str
    tenant_id: str
    model_id: str
    
    # Execution details
    status: SimulationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    
    # Results
    drift_detected: bool = False
    drift_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    drift_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Simulated data
    original_data: Dict[str, Any] = Field(default_factory=dict)
    simulated_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis
    impact_assessment: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    
    # Quarantine simulation
    quarantine_triggered: bool = False
    quarantine_reason: Optional[str] = None

class DriftWhatIfSimulator:
    """Service for simulating drift scenarios and stress conditions"""
    
    def __init__(self):
        self.drift_monitor = DriftBiasMonitor()
        self.logger = logging.getLogger(__name__)
        self.active_simulations: Dict[str, DriftSimulationResult] = {}
    
    async def run_simulation(self, request: DriftSimulationRequest) -> DriftSimulationResult:
        """Run drift simulation scenario"""
        try:
            # Initialize result
            result = DriftSimulationResult(
                simulation_id=request.simulation_id,
                tenant_id=request.tenant_id,
                model_id=request.model_id,
                status=SimulationStatus.RUNNING,
                started_at=datetime.utcnow()
            )
            
            self.active_simulations[request.simulation_id] = result
            
            # Generate simulated data based on scenario
            simulated_data = await self._generate_scenario_data(request)
            result.original_data = request.baseline_data
            result.simulated_data = simulated_data
            
            # Run drift detection on simulated data
            drift_results = await self._run_drift_detection(request, simulated_data)
            result.drift_detected = drift_results["drift_detected"]
            result.drift_alerts = drift_results["alerts"]
            result.drift_scores = drift_results["scores"]
            
            # Assess impact
            impact = await self._assess_impact(request, drift_results)
            result.impact_assessment = impact
            result.recommendations = self._generate_recommendations(request, drift_results, impact)
            
            # Simulate quarantine decision
            quarantine_result = await self._simulate_quarantine(request, drift_results)
            result.quarantine_triggered = quarantine_result["triggered"]
            result.quarantine_reason = quarantine_result.get("reason")
            
            # Complete simulation
            result.status = SimulationStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.execution_time_ms = int((result.completed_at - result.started_at).total_seconds() * 1000)
            
            self.logger.info(f"Completed drift simulation {request.simulation_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Simulation failed {request.simulation_id}: {e}")
            result.status = SimulationStatus.FAILED
            result.completed_at = datetime.utcnow()
            raise
    
    async def _generate_scenario_data(self, request: DriftSimulationRequest) -> Dict[str, List[float]]:
        """Generate data based on stress scenario"""
        simulated_data = {}
        
        for feature_name, baseline_values in request.baseline_data.items():
            if not baseline_values:
                continue
                
            baseline_array = np.array(baseline_values)
            
            if request.scenario == StressScenario.DATA_SHIFT:
                # Shift mean of the distribution
                shift_amount = self._get_severity_multiplier(request.severity) * np.std(baseline_array)
                simulated_values = baseline_array + shift_amount
                
            elif request.scenario == StressScenario.COVARIATE_SHIFT:
                # Change variance/scale
                scale_factor = 1 + (self._get_severity_multiplier(request.severity) * 2)
                simulated_values = baseline_array * scale_factor
                
            elif request.scenario == StressScenario.FEATURE_CORRUPTION:
                # Add noise/corruption
                noise_std = self._get_severity_multiplier(request.severity) * np.std(baseline_array)
                noise = np.random.normal(0, noise_std, len(baseline_array))
                simulated_values = baseline_array + noise
                
            elif request.scenario == StressScenario.SEASONAL_CHANGE:
                # Add seasonal pattern
                time_points = np.linspace(0, 2 * np.pi, len(baseline_array))
                seasonal_amplitude = self._get_severity_multiplier(request.severity) * np.std(baseline_array)
                seasonal_pattern = seasonal_amplitude * np.sin(time_points)
                simulated_values = baseline_array + seasonal_pattern
                
            elif request.scenario == StressScenario.POPULATION_SHIFT:
                # Change distribution shape
                if request.severity == SimulationSeverity.MILD:
                    # Slight skew
                    simulated_values = np.random.beta(2, 5, len(baseline_array)) * np.max(baseline_array)
                elif request.severity == SimulationSeverity.MODERATE:
                    # Bimodal distribution
                    half = len(baseline_array) // 2
                    part1 = np.random.normal(np.mean(baseline_array) - np.std(baseline_array), np.std(baseline_array), half)
                    part2 = np.random.normal(np.mean(baseline_array) + np.std(baseline_array), np.std(baseline_array), len(baseline_array) - half)
                    simulated_values = np.concatenate([part1, part2])
                else:
                    # Extreme: uniform distribution
                    simulated_values = np.random.uniform(np.min(baseline_array), np.max(baseline_array), len(baseline_array))
                    
            elif request.scenario == StressScenario.ADVERSARIAL_ATTACK:
                # Targeted perturbations
                perturbation_strength = self._get_severity_multiplier(request.severity) * 0.1
                perturbations = np.random.choice([-1, 1], len(baseline_array)) * perturbation_strength
                simulated_values = baseline_array * (1 + perturbations)
                
            else:
                # Default: concept drift (gradual shift)
                shift_per_sample = (self._get_severity_multiplier(request.severity) * np.std(baseline_array)) / len(baseline_array)
                shifts = np.cumsum(np.full(len(baseline_array), shift_per_sample))
                simulated_values = baseline_array + shifts
            
            # Apply additional noise if specified
            if request.noise_level > 0:
                noise = np.random.normal(0, request.noise_level * np.std(simulated_values), len(simulated_values))
                simulated_values += noise
            
            # Ensure we have the right sample size
            if len(simulated_values) != request.sample_size:
                if len(simulated_values) > request.sample_size:
                    simulated_values = simulated_values[:request.sample_size]
                else:
                    # Repeat and truncate
                    repeats = (request.sample_size // len(simulated_values)) + 1
                    simulated_values = np.tile(simulated_values, repeats)[:request.sample_size]
            
            simulated_data[feature_name] = simulated_values.tolist()
        
        return simulated_data
    
    async def _run_drift_detection(self, request: DriftSimulationRequest, simulated_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Run drift detection on simulated data"""
        try:
            # Run data drift detection
            drift_alerts = await self.drift_monitor.check_data_drift(
                model_id=request.model_id,
                current_data=simulated_data,
                reference_data=request.baseline_data,
                tenant_id=request.tenant_id,
                workflow_id=f"simulation_{request.simulation_id}",
                step_id="drift_simulation"
            )
            
            # Extract results
            drift_detected = len(drift_alerts) > 0
            alerts_data = []
            scores = {}
            
            for alert in drift_alerts:
                alert_data = {
                    "alert_id": alert.alert_id,
                    "drift_type": alert.drift_type,
                    "severity": alert.severity,
                    "drift_score": alert.drift_score,
                    "threshold": alert.threshold,
                    "affected_features": alert.affected_features,
                    "description": alert.description
                }
                alerts_data.append(alert_data)
                
                # Store scores by feature
                for feature in alert.affected_features:
                    scores[feature] = alert.drift_score
            
            return {
                "drift_detected": drift_detected,
                "alerts": alerts_data,
                "scores": scores
            }
            
        except Exception as e:
            self.logger.error(f"Drift detection failed in simulation: {e}")
            return {
                "drift_detected": False,
                "alerts": [],
                "scores": {}
            }
    
    async def _assess_impact(self, request: DriftSimulationRequest, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of detected drift"""
        impact = {
            "risk_level": "low",
            "confidence_degradation": 0.0,
            "affected_predictions": 0,
            "business_impact": "minimal",
            "urgency": "low"
        }
        
        if drift_results["drift_detected"]:
            # Calculate overall drift severity
            max_drift_score = max(drift_results["scores"].values()) if drift_results["scores"] else 0
            
            if max_drift_score > 0.5:
                impact["risk_level"] = "critical"
                impact["confidence_degradation"] = 0.4
                impact["business_impact"] = "severe"
                impact["urgency"] = "immediate"
            elif max_drift_score > 0.3:
                impact["risk_level"] = "high"
                impact["confidence_degradation"] = 0.25
                impact["business_impact"] = "moderate"
                impact["urgency"] = "high"
            elif max_drift_score > 0.15:
                impact["risk_level"] = "medium"
                impact["confidence_degradation"] = 0.1
                impact["business_impact"] = "low"
                impact["urgency"] = "medium"
            
            # Estimate affected predictions
            total_features = len(request.baseline_data)
            affected_features = len(set().union(*[alert["affected_features"] for alert in drift_results["alerts"]]))
            impact["affected_predictions"] = int((affected_features / total_features) * 100)
        
        # Scenario-specific impact adjustments
        if request.scenario == StressScenario.ADVERSARIAL_ATTACK:
            impact["risk_level"] = "critical"
            impact["urgency"] = "immediate"
            impact["business_impact"] = "security_breach"
        elif request.scenario == StressScenario.POPULATION_SHIFT:
            impact["business_impact"] = "model_obsolescence"
        
        return impact
    
    async def _simulate_quarantine(self, request: DriftSimulationRequest, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate auto-quarantine decision"""
        if not drift_results["drift_detected"]:
            return {"triggered": False}
        
        # Check if any alert has high or critical severity
        critical_alerts = [
            alert for alert in drift_results["alerts"] 
            if alert["severity"] in ["high", "critical"]
        ]
        
        if critical_alerts:
            return {
                "triggered": True,
                "reason": f"Critical drift detected: {critical_alerts[0]['description']}",
                "alert_count": len(critical_alerts)
            }
        
        return {"triggered": False}
    
    def _generate_recommendations(self, request: DriftSimulationRequest, drift_results: Dict[str, Any], impact: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        if not drift_results["drift_detected"]:
            recommendations.append("No drift detected - model appears stable under this scenario")
            return recommendations
        
        # General recommendations
        if impact["risk_level"] in ["high", "critical"]:
            recommendations.append("🚨 URGENT: Consider immediate model retraining or rollback")
            recommendations.append("📊 Implement enhanced monitoring for this scenario")
            
        if impact["risk_level"] in ["medium", "high", "critical"]:
            recommendations.append("🔄 Schedule model retraining with recent data")
            recommendations.append("📈 Increase drift monitoring frequency")
        
        # Scenario-specific recommendations
        if request.scenario == StressScenario.DATA_SHIFT:
            recommendations.append("📋 Review data collection processes for systematic changes")
            recommendations.append("🎯 Consider feature normalization or scaling adjustments")
            
        elif request.scenario == StressScenario.COVARIATE_SHIFT:
            recommendations.append("🔧 Implement domain adaptation techniques")
            recommendations.append("📊 Consider importance weighting for training data")
            
        elif request.scenario == StressScenario.FEATURE_CORRUPTION:
            recommendations.append("🛡️ Implement data quality checks and validation")
            recommendations.append("🔍 Review data pipeline for corruption sources")
            
        elif request.scenario == StressScenario.SEASONAL_CHANGE:
            recommendations.append("📅 Implement seasonal model variants")
            recommendations.append("🔄 Schedule regular model updates aligned with seasons")
            
        elif request.scenario == StressScenario.POPULATION_SHIFT:
            recommendations.append("🎯 Retrain model with representative population data")
            recommendations.append("📊 Consider ensemble methods for diverse populations")
            
        elif request.scenario == StressScenario.ADVERSARIAL_ATTACK:
            recommendations.append("🚨 SECURITY: Implement adversarial defense mechanisms")
            recommendations.append("🛡️ Add input validation and anomaly detection")
            recommendations.append("📋 Review security protocols and access controls")
        
        # Add monitoring recommendations
        recommendations.append("📈 Set up automated alerts for similar drift patterns")
        recommendations.append("🔍 Create custom monitoring rules for this scenario")
        
        return recommendations
    
    def _get_severity_multiplier(self, severity: SimulationSeverity) -> float:
        """Get numeric multiplier for severity level"""
        multipliers = {
            SimulationSeverity.MILD: 0.1,
            SimulationSeverity.MODERATE: 0.3,
            SimulationSeverity.SEVERE: 0.6,
            SimulationSeverity.EXTREME: 1.0
        }
        return multipliers.get(severity, 0.3)

# Global simulator instance
drift_simulator = DriftWhatIfSimulator()

@app.post("/drift-simulation/run", response_model=DriftSimulationResult)
async def run_drift_simulation(request: DriftSimulationRequest, background_tasks: BackgroundTasks):
    """Run drift what-if simulation"""
    return await drift_simulator.run_simulation(request)

@app.get("/drift-simulation/{simulation_id}", response_model=DriftSimulationResult)
async def get_simulation_result(simulation_id: str):
    """Get simulation result by ID"""
    if simulation_id not in drift_simulator.active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return drift_simulator.active_simulations[simulation_id]

@app.get("/drift-simulation/tenant/{tenant_id}")
async def get_tenant_simulations(tenant_id: str, limit: int = 50):
    """Get simulations for a tenant"""
    tenant_simulations = [
        sim for sim in drift_simulator.active_simulations.values()
        if sim.tenant_id == tenant_id
    ]
    
    # Sort by creation time (most recent first)
    tenant_simulations.sort(key=lambda x: x.started_at, reverse=True)
    
    return {
        "simulations": tenant_simulations[:limit],
        "total_count": len(tenant_simulations)
    }

@app.post("/drift-simulation/batch")
async def run_batch_simulations(
    tenant_id: str,
    model_id: str,
    baseline_data: Dict[str, List[float]],
    scenarios: List[StressScenario] = None,
    severities: List[SimulationSeverity] = None,
    created_by: str = "batch_user"
):
    """Run multiple simulations across scenarios and severities"""
    
    if scenarios is None:
        scenarios = list(StressScenario)
    
    if severities is None:
        severities = [SimulationSeverity.MODERATE, SimulationSeverity.SEVERE]
    
    batch_results = []
    
    for scenario in scenarios:
        for severity in severities:
            request = DriftSimulationRequest(
                tenant_id=tenant_id,
                model_id=model_id,
                scenario=scenario,
                severity=severity,
                baseline_data=baseline_data,
                created_by=created_by,
                description=f"Batch simulation: {scenario.value} at {severity.value} level",
                tags=["batch", scenario.value, severity.value]
            )
            
            try:
                result = await drift_simulator.run_simulation(request)
                batch_results.append({
                    "simulation_id": result.simulation_id,
                    "scenario": scenario.value,
                    "severity": severity.value,
                    "status": result.status.value,
                    "drift_detected": result.drift_detected,
                    "quarantine_triggered": result.quarantine_triggered
                })
            except Exception as e:
                batch_results.append({
                    "simulation_id": request.simulation_id,
                    "scenario": scenario.value,
                    "severity": severity.value,
                    "status": "failed",
                    "error": str(e)
                })
    
    return {
        "batch_id": str(uuid.uuid4()),
        "total_simulations": len(batch_results),
        "results": batch_results
    }

@app.get("/drift-simulation/scenarios")
async def get_available_scenarios():
    """Get available stress scenarios and their descriptions"""
    scenarios = {
        StressScenario.DATA_SHIFT: "Mean shift in feature distributions",
        StressScenario.CONCEPT_DRIFT: "Gradual change in underlying relationships",
        StressScenario.COVARIATE_SHIFT: "Change in feature variance/scale",
        StressScenario.LABEL_SHIFT: "Change in target variable distribution",
        StressScenario.FEATURE_CORRUPTION: "Noise and data quality issues",
        StressScenario.SEASONAL_CHANGE: "Periodic/seasonal variations",
        StressScenario.POPULATION_SHIFT: "Change in population characteristics",
        StressScenario.ADVERSARIAL_ATTACK: "Targeted adversarial perturbations"
    }
    
    return {
        "scenarios": [
            {
                "name": scenario.value,
                "description": description,
                "severity_levels": [s.value for s in SimulationSeverity]
            }
            for scenario, description in scenarios.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
