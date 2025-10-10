"""
Task 6.3.60: Golden datasets per model for health checks
Create golden datasets per model for constant baselines and drift detection
"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class GoldenRecord:
    """Single golden dataset record"""
    record_id: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class GoldenDataset:
    """Golden dataset for a model"""
    dataset_id: str
    model_id: str
    model_version: str
    records: List[GoldenRecord]
    dataset_hash: str
    created_at: datetime
    description: str
    tags: List[str]

@dataclass
class HealthCheckResult:
    """Health check result using golden dataset"""
    model_id: str
    dataset_id: str
    total_records: int
    passed_records: int
    failed_records: int
    accuracy: float
    drift_score: float
    execution_time_ms: int
    timestamp: datetime
    failures: List[Dict[str, Any]]

class GoldenDatasetService:
    """
    Golden dataset service for model health checks
    Task 6.3.60: Constant baselines and drift detection reference
    """
    
    def __init__(self, storage_path: str = "golden_datasets"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.datasets: Dict[str, GoldenDataset] = {}
        self.health_check_results: Dict[str, List[HealthCheckResult]] = {}
    
    def create_golden_dataset(
        self,
        model_id: str,
        model_version: str,
        records: List[Dict[str, Any]],
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """Create a golden dataset for a model"""
        try:
            dataset_id = f"{model_id}_{model_version}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create golden records
            golden_records = []
            for i, record in enumerate(records):
                golden_record = GoldenRecord(
                    record_id=f"{dataset_id}_{i}",
                    input_data=record["input"],
                    expected_output=record["expected_output"],
                    metadata=record.get("metadata", {}),
                    created_at=datetime.utcnow()
                )
                golden_records.append(golden_record)
            
            # Calculate dataset hash for integrity
            dataset_content = json.dumps([asdict(r) for r in golden_records], sort_keys=True)
            dataset_hash = hashlib.sha256(dataset_content.encode()).hexdigest()
            
            # Create dataset
            golden_dataset = GoldenDataset(
                dataset_id=dataset_id,
                model_id=model_id,
                model_version=model_version,
                records=golden_records,
                dataset_hash=dataset_hash,
                created_at=datetime.utcnow(),
                description=description,
                tags=tags or []
            )
            
            # Store dataset
            self.datasets[dataset_id] = golden_dataset
            self._save_dataset_to_file(golden_dataset)
            
            logger.info(f"Created golden dataset: {dataset_id} with {len(golden_records)} records")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Failed to create golden dataset: {e}")
            raise
    
    def load_golden_dataset(self, dataset_id: str) -> Optional[GoldenDataset]:
        """Load golden dataset from storage"""
        if dataset_id in self.datasets:
            return self.datasets[dataset_id]
        
        # Try loading from file
        dataset_file = self.storage_path / f"{dataset_id}.json"
        if dataset_file.exists():
            try:
                with open(dataset_file, 'r') as f:
                    dataset_data = json.load(f)
                
                # Reconstruct dataset object
                records = [
                    GoldenRecord(
                        record_id=r["record_id"],
                        input_data=r["input_data"],
                        expected_output=r["expected_output"],
                        metadata=r["metadata"],
                        created_at=datetime.fromisoformat(r["created_at"])
                    )
                    for r in dataset_data["records"]
                ]
                
                dataset = GoldenDataset(
                    dataset_id=dataset_data["dataset_id"],
                    model_id=dataset_data["model_id"],
                    model_version=dataset_data["model_version"],
                    records=records,
                    dataset_hash=dataset_data["dataset_hash"],
                    created_at=datetime.fromisoformat(dataset_data["created_at"]),
                    description=dataset_data["description"],
                    tags=dataset_data["tags"]
                )
                
                self.datasets[dataset_id] = dataset
                return dataset
                
            except Exception as e:
                logger.error(f"Failed to load golden dataset {dataset_id}: {e}")
        
        return None
    
    async def run_health_check(self, model_id: str, dataset_id: str, tolerance: float = 0.1) -> HealthCheckResult:
        """Run health check using golden dataset"""
        dataset = self.load_golden_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Golden dataset not found: {dataset_id}")
        
        if dataset.model_id != model_id:
            raise ValueError(f"Dataset {dataset_id} is not for model {model_id}")
        
        start_time = datetime.utcnow()
        passed_records = 0
        failed_records = 0
        failures = []
        drift_scores = []
        
        # Run inference on each golden record
        for record in dataset.records:
            try:
                # Get model prediction
                actual_output = await self._invoke_model(model_id, record.input_data)
                
                # Compare with expected output
                is_match, drift_score = self._compare_outputs(
                    actual_output,
                    record.expected_output,
                    tolerance
                )
                
                drift_scores.append(drift_score)
                
                if is_match:
                    passed_records += 1
                else:
                    failed_records += 1
                    failures.append({
                        "record_id": record.record_id,
                        "expected": record.expected_output,
                        "actual": actual_output,
                        "drift_score": drift_score
                    })
            
            except Exception as e:
                failed_records += 1
                failures.append({
                    "record_id": record.record_id,
                    "error": str(e)
                })
        
        # Calculate metrics
        total_records = len(dataset.records)
        accuracy = passed_records / total_records if total_records > 0 else 0.0
        avg_drift_score = sum(drift_scores) / len(drift_scores) if drift_scores else 0.0
        execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Create result
        result = HealthCheckResult(
            model_id=model_id,
            dataset_id=dataset_id,
            total_records=total_records,
            passed_records=passed_records,
            failed_records=failed_records,
            accuracy=accuracy,
            drift_score=avg_drift_score,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.utcnow(),
            failures=failures
        )
        
        # Store result
        if model_id not in self.health_check_results:
            self.health_check_results[model_id] = []
        self.health_check_results[model_id].append(result)
        
        logger.info(f"Health check completed: {accuracy:.2%} accuracy, {avg_drift_score:.3f} drift score")
        return result
    
    async def _invoke_model(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke model for health check"""
        # Simulate model inference
        return {
            "prediction": 0.75,
            "confidence": 0.85,
            "probabilities": [0.25, 0.75]
        }
    
    def _compare_outputs(
        self,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
        tolerance: float
    ) -> tuple[bool, float]:
        """Compare actual vs expected outputs"""
        drift_score = 0.0
        total_comparisons = 0
        
        for key, expected_value in expected.items():
            if key not in actual:
                drift_score += 1.0  # Missing key is maximum drift
                total_comparisons += 1
                continue
            
            actual_value = actual[key]
            
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                # Numerical comparison
                diff = abs(actual_value - expected_value)
                if expected_value != 0:
                    relative_diff = diff / abs(expected_value)
                else:
                    relative_diff = diff
                drift_score += min(relative_diff, 1.0)
            else:
                # Exact match for non-numerical values
                if actual_value != expected_value:
                    drift_score += 1.0
            
            total_comparisons += 1
        
        avg_drift_score = drift_score / total_comparisons if total_comparisons > 0 else 0.0
        is_match = avg_drift_score <= tolerance
        
        return is_match, avg_drift_score
    
    def _save_dataset_to_file(self, dataset: GoldenDataset) -> None:
        """Save dataset to file"""
        dataset_file = self.storage_path / f"{dataset.dataset_id}.json"
        
        # Convert to serializable format
        dataset_data = {
            "dataset_id": dataset.dataset_id,
            "model_id": dataset.model_id,
            "model_version": dataset.model_version,
            "dataset_hash": dataset.dataset_hash,
            "created_at": dataset.created_at.isoformat(),
            "description": dataset.description,
            "tags": dataset.tags,
            "records": [
                {
                    "record_id": r.record_id,
                    "input_data": r.input_data,
                    "expected_output": r.expected_output,
                    "metadata": r.metadata,
                    "created_at": r.created_at.isoformat()
                }
                for r in dataset.records
            ]
        }
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f, indent=2)
    
    def get_datasets_for_model(self, model_id: str) -> List[str]:
        """Get all dataset IDs for a model"""
        return [
            dataset_id for dataset_id, dataset in self.datasets.items()
            if dataset.model_id == model_id
        ]
    
    def get_health_check_history(self, model_id: str) -> List[HealthCheckResult]:
        """Get health check history for a model"""
        return self.health_check_results.get(model_id, [])

# Global golden dataset service instance
golden_dataset_service = GoldenDatasetService()
