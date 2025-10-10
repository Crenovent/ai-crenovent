"""
Task 4.4.24: Add tenant-level benchmarking (compare trust scores across tenants)
- Adoption gamification
- Metrics warehouse cross-tenant analysis
- Optional shared reports
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import statistics

app = FastAPI(title="RBIA Tenant-Level Benchmarking")
logger = logging.getLogger(__name__)

class BenchmarkMetric(str, Enum):
    TRUST_SCORE = "trust_score"
    ACCURACY = "accuracy"
    ADOPTION_RATE = "adoption_rate"
    ROI = "roi"
    SLA_COMPLIANCE = "sla_compliance"
    INCIDENT_FREQUENCY = "incident_frequency"

class TenantTier(str, Enum):
    ENTERPRISE = "enterprise"
    MID_MARKET = "mid_market"
    SMB = "smb"

class TenantBenchmark(BaseModel):
    benchmark_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    tenant_name: str
    tenant_tier: TenantTier
    
    # Benchmark metrics
    trust_score: float = 0.0
    accuracy: float = 0.0
    adoption_rate: float = 0.0
    roi_percentage: float = 0.0
    sla_compliance: float = 0.0
    incident_frequency: float = 0.0
    
    # Percentile rankings
    trust_score_percentile: float = 0.0
    accuracy_percentile: float = 0.0
    adoption_percentile: float = 0.0
    roi_percentile: float = 0.0
    
    # Overall ranking
    overall_rank: int = 0
    total_tenants: int = 0
    
    measurement_date: datetime = Field(default_factory=datetime.utcnow)

class BenchmarkReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Report metadata
    report_title: str = "Tenant Benchmarking Report"
    report_period: str = "monthly"
    total_tenants: int = 0
    
    # Tenant rankings
    top_performers: List[TenantBenchmark] = Field(default_factory=list)
    bottom_performers: List[TenantBenchmark] = Field(default_factory=list)
    
    # Tier comparisons
    enterprise_avg: Dict[str, float] = Field(default_factory=dict)
    mid_market_avg: Dict[str, float] = Field(default_factory=dict)
    smb_avg: Dict[str, float] = Field(default_factory=dict)
    
    # Distribution stats
    metric_distributions: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
tenant_benchmarks_store: Dict[str, TenantBenchmark] = {}
benchmark_reports_store: Dict[str, BenchmarkReport] = {}

class TenantBenchmarkingEngine:
    def __init__(self):
        # Mock tenant data (in production, would come from actual metrics services)
        self.mock_tenant_data = {
            "tenant_1": {
                "name": "Acme Corp",
                "tier": TenantTier.ENTERPRISE,
                "trust_score": 0.89,
                "accuracy": 92.5,
                "adoption_rate": 85.3,
                "roi_percentage": 245.7,
                "sla_compliance": 98.2,
                "incident_frequency": 2.1
            },
            "tenant_2": {
                "name": "Beta Industries", 
                "tier": TenantTier.MID_MARKET,
                "trust_score": 0.82,
                "accuracy": 87.1,
                "adoption_rate": 72.8,
                "roi_percentage": 189.3,
                "sla_compliance": 95.7,
                "incident_frequency": 3.8
            },
            "tenant_3": {
                "name": "Gamma Solutions",
                "tier": TenantTier.SMB,
                "trust_score": 0.76,
                "accuracy": 83.2,
                "adoption_rate": 68.5,
                "roi_percentage": 156.9,
                "sla_compliance": 92.1,
                "incident_frequency": 5.2
            },
            "tenant_4": {
                "name": "Delta Enterprises",
                "tier": TenantTier.ENTERPRISE,
                "trust_score": 0.91,
                "accuracy": 94.8,
                "adoption_rate": 89.7,
                "roi_percentage": 278.4,
                "sla_compliance": 99.1,
                "incident_frequency": 1.6
            },
            "tenant_5": {
                "name": "Echo Systems",
                "tier": TenantTier.MID_MARKET,
                "trust_score": 0.78,
                "accuracy": 85.6,
                "adoption_rate": 75.2,
                "roi_percentage": 198.7,
                "sla_compliance": 94.3,
                "incident_frequency": 4.1
            }
        }
    
    def calculate_percentiles(self, values: List[float], target_value: float) -> float:
        """Calculate percentile ranking for a value"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        rank = sum(1 for v in sorted_values if v <= target_value)
        percentile = (rank / len(sorted_values)) * 100
        
        return round(percentile, 1)
    
    def generate_tenant_benchmarks(self) -> List[TenantBenchmark]:
        """Generate benchmarks for all tenants"""
        
        benchmarks = []
        
        # Collect all metric values for percentile calculations
        all_trust_scores = [data["trust_score"] for data in self.mock_tenant_data.values()]
        all_accuracy = [data["accuracy"] for data in self.mock_tenant_data.values()]
        all_adoption = [data["adoption_rate"] for data in self.mock_tenant_data.values()]
        all_roi = [data["roi_percentage"] for data in self.mock_tenant_data.values()]
        
        # Generate benchmarks for each tenant
        for tenant_id, data in self.mock_tenant_data.items():
            
            # Calculate percentiles
            trust_percentile = self.calculate_percentiles(all_trust_scores, data["trust_score"])
            accuracy_percentile = self.calculate_percentiles(all_accuracy, data["accuracy"])
            adoption_percentile = self.calculate_percentiles(all_adoption, data["adoption_rate"])
            roi_percentile = self.calculate_percentiles(all_roi, data["roi_percentage"])
            
            benchmark = TenantBenchmark(
                tenant_id=tenant_id,
                tenant_name=data["name"],
                tenant_tier=data["tier"],
                trust_score=data["trust_score"],
                accuracy=data["accuracy"],
                adoption_rate=data["adoption_rate"],
                roi_percentage=data["roi_percentage"],
                sla_compliance=data["sla_compliance"],
                incident_frequency=data["incident_frequency"],
                trust_score_percentile=trust_percentile,
                accuracy_percentile=accuracy_percentile,
                adoption_percentile=adoption_percentile,
                roi_percentile=roi_percentile,
                total_tenants=len(self.mock_tenant_data)
            )
            
            benchmarks.append(benchmark)
        
        # Calculate overall rankings based on composite score
        for i, benchmark in enumerate(benchmarks):
            composite_score = (
                benchmark.trust_score_percentile * 0.3 +
                benchmark.accuracy_percentile * 0.25 +
                benchmark.adoption_percentile * 0.25 +
                benchmark.roi_percentile * 0.20
            )
            benchmark.metadata = {"composite_score": composite_score}
        
        # Sort by composite score and assign ranks
        benchmarks.sort(key=lambda x: x.metadata["composite_score"], reverse=True)
        for i, benchmark in enumerate(benchmarks):
            benchmark.overall_rank = i + 1
        
        return benchmarks
    
    def generate_benchmark_report(self) -> BenchmarkReport:
        """Generate comprehensive benchmarking report"""
        
        benchmarks = self.generate_tenant_benchmarks()
        
        # Store benchmarks
        for benchmark in benchmarks:
            tenant_benchmarks_store[benchmark.benchmark_id] = benchmark
        
        # Get top and bottom performers
        top_performers = benchmarks[:3]  # Top 3
        bottom_performers = benchmarks[-2:]  # Bottom 2
        
        # Calculate tier averages
        enterprise_tenants = [b for b in benchmarks if b.tenant_tier == TenantTier.ENTERPRISE]
        mid_market_tenants = [b for b in benchmarks if b.tenant_tier == TenantTier.MID_MARKET]
        smb_tenants = [b for b in benchmarks if b.tenant_tier == TenantTier.SMB]
        
        enterprise_avg = self._calculate_tier_averages(enterprise_tenants)
        mid_market_avg = self._calculate_tier_averages(mid_market_tenants)
        smb_avg = self._calculate_tier_averages(smb_tenants)
        
        # Calculate metric distributions
        metric_distributions = {
            "trust_score": {
                "min": min(b.trust_score for b in benchmarks),
                "max": max(b.trust_score for b in benchmarks),
                "mean": statistics.mean([b.trust_score for b in benchmarks]),
                "median": statistics.median([b.trust_score for b in benchmarks]),
                "std_dev": statistics.stdev([b.trust_score for b in benchmarks])
            },
            "accuracy": {
                "min": min(b.accuracy for b in benchmarks),
                "max": max(b.accuracy for b in benchmarks),
                "mean": statistics.mean([b.accuracy for b in benchmarks]),
                "median": statistics.median([b.accuracy for b in benchmarks]),
                "std_dev": statistics.stdev([b.accuracy for b in benchmarks])
            },
            "roi_percentage": {
                "min": min(b.roi_percentage for b in benchmarks),
                "max": max(b.roi_percentage for b in benchmarks),
                "mean": statistics.mean([b.roi_percentage for b in benchmarks]),
                "median": statistics.median([b.roi_percentage for b in benchmarks]),
                "std_dev": statistics.stdev([b.roi_percentage for b in benchmarks])
            }
        }
        
        return BenchmarkReport(
            total_tenants=len(benchmarks),
            top_performers=top_performers,
            bottom_performers=bottom_performers,
            enterprise_avg=enterprise_avg,
            mid_market_avg=mid_market_avg,
            smb_avg=smb_avg,
            metric_distributions=metric_distributions
        )
    
    def _calculate_tier_averages(self, tier_tenants: List[TenantBenchmark]) -> Dict[str, float]:
        """Calculate average metrics for a tenant tier"""
        
        if not tier_tenants:
            return {}
        
        return {
            "trust_score": round(statistics.mean([t.trust_score for t in tier_tenants]), 3),
            "accuracy": round(statistics.mean([t.accuracy for t in tier_tenants]), 1),
            "adoption_rate": round(statistics.mean([t.adoption_rate for t in tier_tenants]), 1),
            "roi_percentage": round(statistics.mean([t.roi_percentage for t in tier_tenants]), 1),
            "sla_compliance": round(statistics.mean([t.sla_compliance for t in tier_tenants]), 1)
        }

# Global benchmarking engine
benchmarking_engine = TenantBenchmarkingEngine()

@app.post("/tenant-benchmarking/generate", response_model=BenchmarkReport)
async def generate_benchmarking_report():
    """Generate tenant benchmarking report"""
    
    report = benchmarking_engine.generate_benchmark_report()
    
    # Store report
    benchmark_reports_store[report.report_id] = report
    
    logger.info(f"✅ Generated tenant benchmarking report with {report.total_tenants} tenants")
    return report

@app.get("/tenant-benchmarking/tenant/{tenant_id}")
async def get_tenant_benchmark(tenant_id: str):
    """Get benchmark data for a specific tenant"""
    
    tenant_benchmark = next(
        (benchmark for benchmark in tenant_benchmarks_store.values()
         if benchmark.tenant_id == tenant_id),
        None
    )
    
    if not tenant_benchmark:
        raise HTTPException(status_code=404, detail="Tenant benchmark not found")
    
    return tenant_benchmark

@app.get("/tenant-benchmarking/rankings")
async def get_tenant_rankings(metric: BenchmarkMetric = BenchmarkMetric.TRUST_SCORE):
    """Get tenant rankings by specific metric"""
    
    benchmarks = list(tenant_benchmarks_store.values())
    
    if not benchmarks:
        raise HTTPException(status_code=404, detail="No benchmark data available")
    
    # Sort by specified metric
    if metric == BenchmarkMetric.TRUST_SCORE:
        benchmarks.sort(key=lambda x: x.trust_score, reverse=True)
    elif metric == BenchmarkMetric.ACCURACY:
        benchmarks.sort(key=lambda x: x.accuracy, reverse=True)
    elif metric == BenchmarkMetric.ADOPTION_RATE:
        benchmarks.sort(key=lambda x: x.adoption_rate, reverse=True)
    elif metric == BenchmarkMetric.ROI:
        benchmarks.sort(key=lambda x: x.roi_percentage, reverse=True)
    elif metric == BenchmarkMetric.SLA_COMPLIANCE:
        benchmarks.sort(key=lambda x: x.sla_compliance, reverse=True)
    elif metric == BenchmarkMetric.INCIDENT_FREQUENCY:
        benchmarks.sort(key=lambda x: x.incident_frequency)  # Lower is better
    
    return {
        "metric": metric.value,
        "rankings": [
            {
                "rank": i + 1,
                "tenant_name": benchmark.tenant_name,
                "tenant_tier": benchmark.tenant_tier,
                "metric_value": getattr(benchmark, metric.value),
                "percentile": getattr(benchmark, f"{metric.value}_percentile", 0)
            }
            for i, benchmark in enumerate(benchmarks)
        ]
    }

@app.get("/tenant-benchmarking/tier-comparison")
async def get_tier_comparison():
    """Get comparison across tenant tiers"""
    
    latest_report = max(
        benchmark_reports_store.values(),
        key=lambda x: x.generated_at,
        default=None
    )
    
    if not latest_report:
        raise HTTPException(status_code=404, detail="No benchmark reports available")
    
    return {
        "enterprise": latest_report.enterprise_avg,
        "mid_market": latest_report.mid_market_avg,
        "smb": latest_report.smb_avg,
        "comparison_insights": [
            "Enterprise tenants show highest trust scores and ROI",
            "Mid-market tenants have balanced performance across metrics",
            "SMB tenants have room for improvement in adoption rates"
        ]
    }

@app.get("/tenant-benchmarking/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Get tenant leaderboard"""
    
    benchmarks = list(tenant_benchmarks_store.values())
    
    if not benchmarks:
        return {"leaderboard": [], "total_tenants": 0}
    
    # Sort by overall rank
    benchmarks.sort(key=lambda x: x.overall_rank)
    leaderboard = benchmarks[:limit]
    
    return {
        "leaderboard": [
            {
                "rank": benchmark.overall_rank,
                "tenant_name": benchmark.tenant_name,
                "tenant_tier": benchmark.tenant_tier,
                "trust_score": benchmark.trust_score,
                "accuracy": benchmark.accuracy,
                "roi_percentage": benchmark.roi_percentage,
                "composite_score": benchmark.metadata.get("composite_score", 0)
            }
            for benchmark in leaderboard
        ],
        "total_tenants": len(benchmarks)
    }

@app.get("/tenant-benchmarking/summary")
async def get_benchmarking_summary():
    """Get benchmarking service summary"""
    
    total_benchmarks = len(tenant_benchmarks_store)
    total_reports = len(benchmark_reports_store)
    
    if total_benchmarks == 0:
        return {
            "total_tenants": 0,
            "total_reports": 0,
            "tier_distribution": {}
        }
    
    # Tier distribution
    tier_counts = {}
    for benchmark in tenant_benchmarks_store.values():
        tier = benchmark.tenant_tier.value
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    return {
        "total_tenants": len(set(b.tenant_id for b in tenant_benchmarks_store.values())),
        "total_reports": total_reports,
        "tier_distribution": tier_counts,
        "latest_report_date": max(
            [r.generated_at for r in benchmark_reports_store.values()],
            default=datetime.utcnow()
        ).isoformat() if benchmark_reports_store else None
    }
