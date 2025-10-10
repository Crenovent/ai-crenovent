"""
Task 3.2.32: Roll-forward reproducibility (seed/data snapshot, env pinning)
- Repeatable outcomes
- Registry + artifacts
- Deterministic builds
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import json
import hashlib
import logging
import os
import zipfile
import io
from dataclasses import dataclass

app = FastAPI(title="RBIA Roll-Forward Reproducibility Service")
logger = logging.getLogger(__name__)

class ArtifactType(str, Enum):
    MODEL_CHECKPOINT = "model_checkpoint"
    TRAINING_DATA = "training_data"
    VALIDATION_DATA = "validation_data"
    FEATURE_DEFINITIONS = "feature_definitions"
    PREPROCESSING_PIPELINE = "preprocessing_pipeline"
    ENVIRONMENT_SPEC = "environment_spec"
    CONFIGURATION = "configuration"
    RANDOM_SEEDS = "random_seeds"

class EnvironmentType(str, Enum):
    DOCKER = "docker"
    CONDA = "conda"
    VIRTUALENV = "virtualenv"
    REQUIREMENTS_TXT = "requirements_txt"

class ReproducibilityStatus(str, Enum):
    PENDING = "pending"
    CAPTURING = "capturing"
    CAPTURED = "captured"
    VERIFIED = "verified"
    FAILED = "failed"

class DataSnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = Field(..., description="Name of the dataset")
    version: str = Field(..., description="Dataset version")
    row_count: int = Field(..., description="Number of rows in dataset")
    column_count: int = Field(..., description="Number of columns in dataset")
    data_hash: str = Field(..., description="Hash of the dataset content")
    schema_hash: str = Field(..., description="Hash of the dataset schema")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    storage_path: str = Field(..., description="Path to stored snapshot")
    compression_type: Optional[str] = Field(None, description="Compression used")
    file_size_bytes: int = Field(..., description="Size of snapshot file")

class EnvironmentSpec(BaseModel):
    spec_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    environment_type: EnvironmentType = Field(..., description="Type of environment specification")
    python_version: str = Field(..., description="Python version")
    packages: Dict[str, str] = Field(..., description="Package name to version mapping")
    system_packages: Optional[Dict[str, str]] = Field(None, description="System packages")
    environment_variables: Dict[str, str] = Field(default={}, description="Required environment variables")
    cuda_version: Optional[str] = Field(None, description="CUDA version if applicable")
    hardware_requirements: Dict[str, Any] = Field(default={}, description="Hardware requirements")
    dockerfile_content: Optional[str] = Field(None, description="Dockerfile content if Docker environment")
    conda_yaml: Optional[str] = Field(None, description="Conda environment YAML")
    requirements_txt: Optional[str] = Field(None, description="Requirements.txt content")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    environment_hash: str = Field(..., description="Hash of environment specification")

class RandomSeedConfig(BaseModel):
    numpy_seed: int = Field(..., description="NumPy random seed")
    python_seed: int = Field(..., description="Python random seed")
    torch_seed: Optional[int] = Field(None, description="PyTorch random seed")
    tensorflow_seed: Optional[int] = Field(None, description="TensorFlow random seed")
    sklearn_seed: Optional[int] = Field(None, description="Scikit-learn random seed")
    custom_seeds: Dict[str, int] = Field(default={}, description="Custom library seeds")

class ReproducibilityArtifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    name: str = Field(..., description="Artifact name")
    version: str = Field(..., description="Artifact version")
    file_path: str = Field(..., description="Path to artifact file")
    file_hash: str = Field(..., description="Hash of artifact file")
    file_size_bytes: int = Field(..., description="Size of artifact file")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    dependencies: List[str] = Field(default=[], description="Dependent artifact IDs")

class ReproducibilityPackage(BaseModel):
    package_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="User who created the package")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Core reproducibility components
    data_snapshots: List[DataSnapshot] = Field(default=[], description="Data snapshots")
    environment_spec: EnvironmentSpec = Field(..., description="Environment specification")
    random_seeds: RandomSeedConfig = Field(..., description="Random seed configuration")
    artifacts: List[ReproducibilityArtifact] = Field(default=[], description="Model artifacts")
    
    # Reproducibility metadata
    status: ReproducibilityStatus = ReproducibilityStatus.PENDING
    verification_results: Optional[Dict[str, Any]] = Field(None, description="Verification test results")
    package_hash: Optional[str] = Field(None, description="Hash of entire package")
    storage_path: Optional[str] = Field(None, description="Path to packaged artifacts")
    
    # Configuration
    configuration: Dict[str, Any] = Field(default={}, description="Model training/inference configuration")
    hyperparameters: Dict[str, Any] = Field(default={}, description="Model hyperparameters")
    
    # Validation
    validation_metrics: Dict[str, float] = Field(default={}, description="Validation metrics from original run")
    reproduction_tolerance: float = Field(default=1e-6, description="Tolerance for reproduction validation")

class ReproducibilityRequest(BaseModel):
    model_id: str = Field(..., description="Model to create reproducibility package for")
    model_version: str = Field(..., description="Model version")
    tenant_id: str = Field(..., description="Tenant identifier")
    include_training_data: bool = Field(default=True, description="Include training data snapshot")
    include_validation_data: bool = Field(default=True, description="Include validation data snapshot")
    include_test_data: bool = Field(default=False, description="Include test data snapshot")
    data_compression: bool = Field(default=True, description="Compress data snapshots")
    environment_type: EnvironmentType = Field(default=EnvironmentType.DOCKER, description="Environment specification type")
    custom_seeds: Dict[str, int] = Field(default={}, description="Custom random seeds")

class ReproductionRequest(BaseModel):
    package_id: str = Field(..., description="Reproducibility package ID")
    validation_mode: str = Field(default="strict", description="Validation mode: strict, moderate, or lenient")
    run_full_validation: bool = Field(default=True, description="Run full validation tests")
    target_environment: Optional[str] = Field(None, description="Target environment for reproduction")

# In-memory storage for reproducibility packages (replace with actual database)
reproducibility_packages: Dict[str, ReproducibilityPackage] = {}

@app.post("/reproducibility/capture", response_model=ReproducibilityPackage)
async def capture_reproducibility_package(request: ReproducibilityRequest):
    """
    Capture a complete reproducibility package for a model
    """
    package = ReproducibilityPackage(
        model_id=request.model_id,
        model_version=request.model_version,
        tenant_id=request.tenant_id,
        created_by="system",  # In production, get from auth context
        status=ReproducibilityStatus.CAPTURING
    )
    
    try:
        # Capture data snapshots
        if request.include_training_data:
            training_snapshot = await capture_data_snapshot(
                "training_data", 
                request.model_id, 
                request.data_compression
            )
            package.data_snapshots.append(training_snapshot)
        
        if request.include_validation_data:
            validation_snapshot = await capture_data_snapshot(
                "validation_data", 
                request.model_id, 
                request.data_compression
            )
            package.data_snapshots.append(validation_snapshot)
        
        if request.include_test_data:
            test_snapshot = await capture_data_snapshot(
                "test_data", 
                request.model_id, 
                request.data_compression
            )
            package.data_snapshots.append(test_snapshot)
        
        # Capture environment specification
        package.environment_spec = await capture_environment_spec(request.environment_type)
        
        # Capture random seeds
        package.random_seeds = capture_random_seeds(request.custom_seeds)
        
        # Capture model artifacts
        package.artifacts = await capture_model_artifacts(request.model_id, request.model_version)
        
        # Capture configuration
        package.configuration = await capture_model_configuration(request.model_id)
        package.hyperparameters = await capture_hyperparameters(request.model_id)
        
        # Calculate package hash for integrity
        package.package_hash = calculate_package_hash(package)
        
        package.status = ReproducibilityStatus.CAPTURED
        reproducibility_packages[package.package_id] = package
        
        logger.info(f"Reproducibility package captured: {package.package_id} for model {request.model_id}")
        
        return package
        
    except Exception as e:
        logger.error(f"Failed to capture reproducibility package: {e}")
        package.status = ReproducibilityStatus.FAILED
        reproducibility_packages[package.package_id] = package
        raise HTTPException(status_code=500, detail=f"Failed to capture reproducibility package: {e}")

async def capture_data_snapshot(dataset_name: str, model_id: str, compress: bool) -> DataSnapshot:
    """Capture a snapshot of dataset"""
    # Simulate data snapshot capture
    # In production, this would read actual dataset and create snapshot
    
    # Generate mock data characteristics
    row_count = 10000 + hash(model_id + dataset_name) % 90000  # Deterministic but varied
    column_count = 10 + hash(dataset_name) % 40
    
    # Simulate data content hash
    data_content = f"{model_id}_{dataset_name}_{row_count}_{column_count}"
    data_hash = hashlib.sha256(data_content.encode()).hexdigest()
    
    # Simulate schema hash
    schema_content = f"schema_{dataset_name}_{column_count}"
    schema_hash = hashlib.sha256(schema_content.encode()).hexdigest()
    
    # Simulate file storage
    storage_path = f"/reproducibility/data/{model_id}/{dataset_name}_{data_hash[:8]}.parquet"
    if compress:
        storage_path += ".gz"
    
    file_size = row_count * column_count * 8  # Simulate file size
    if compress:
        file_size = int(file_size * 0.3)  # Simulate compression
    
    return DataSnapshot(
        dataset_name=dataset_name,
        version="1.0",
        row_count=row_count,
        column_count=column_count,
        data_hash=data_hash,
        schema_hash=schema_hash,
        storage_path=storage_path,
        compression_type="gzip" if compress else None,
        file_size_bytes=file_size
    )

async def capture_environment_spec(env_type: EnvironmentType) -> EnvironmentSpec:
    """Capture current environment specification"""
    
    # Simulate current environment capture
    packages = {
        "numpy": "1.24.3",
        "pandas": "2.0.3",
        "scikit-learn": "1.3.0",
        "torch": "2.0.1",
        "transformers": "4.30.2",
        "fastapi": "0.100.0",
        "pydantic": "2.0.3"
    }
    
    system_packages = {
        "gcc": "9.4.0",
        "cmake": "3.20.0",
        "git": "2.34.1"
    }
    
    env_variables = {
        "PYTHONPATH": "/app",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "4"
    }
    
    hardware_reqs = {
        "min_memory_gb": 8,
        "recommended_memory_gb": 16,
        "gpu_required": True,
        "min_gpu_memory_gb": 4
    }
    
    # Generate environment-specific content
    dockerfile_content = None
    conda_yaml = None
    requirements_txt = None
    
    if env_type == EnvironmentType.DOCKER:
        dockerfile_content = f"""
FROM python:3.9-slim

# Install system packages
RUN apt-get update && apt-get install -y gcc cmake git

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4

# Install Python packages
RUN pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0

WORKDIR /app
"""
    
    elif env_type == EnvironmentType.CONDA:
        conda_yaml = """
name: rbia-reproducible
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - numpy=1.24.3
  - pandas=2.0.3
  - scikit-learn=1.3.0
  - pytorch=2.0.1
  - pip
  - pip:
    - transformers==4.30.2
    - fastapi==0.100.0
"""
    
    elif env_type == EnvironmentType.REQUIREMENTS_TXT:
        requirements_txt = "\n".join([f"{pkg}=={ver}" for pkg, ver in packages.items()])
    
    # Calculate environment hash
    env_content = json.dumps({
        "python_version": "3.9.7",
        "packages": packages,
        "system_packages": system_packages,
        "environment_variables": env_variables
    }, sort_keys=True)
    env_hash = hashlib.sha256(env_content.encode()).hexdigest()
    
    return EnvironmentSpec(
        environment_type=env_type,
        python_version="3.9.7",
        packages=packages,
        system_packages=system_packages,
        environment_variables=env_variables,
        cuda_version="11.8" if hardware_reqs["gpu_required"] else None,
        hardware_requirements=hardware_reqs,
        dockerfile_content=dockerfile_content,
        conda_yaml=conda_yaml,
        requirements_txt=requirements_txt,
        environment_hash=env_hash
    )

def capture_random_seeds(custom_seeds: Dict[str, int]) -> RandomSeedConfig:
    """Capture random seed configuration"""
    
    # Generate deterministic seeds based on current time
    base_seed = int(datetime.utcnow().timestamp()) % 2**31
    
    return RandomSeedConfig(
        numpy_seed=base_seed,
        python_seed=base_seed + 1,
        torch_seed=base_seed + 2,
        tensorflow_seed=base_seed + 3,
        sklearn_seed=base_seed + 4,
        custom_seeds=custom_seeds
    )

async def capture_model_artifacts(model_id: str, model_version: str) -> List[ReproducibilityArtifact]:
    """Capture model artifacts"""
    artifacts = []
    
    # Model checkpoint
    checkpoint_artifact = ReproducibilityArtifact(
        artifact_type=ArtifactType.MODEL_CHECKPOINT,
        name=f"{model_id}_checkpoint",
        version=model_version,
        file_path=f"/models/{model_id}/checkpoint_{model_version}.pt",
        file_hash=hashlib.sha256(f"{model_id}_{model_version}_checkpoint".encode()).hexdigest(),
        file_size_bytes=1024 * 1024 * 50,  # 50MB
        metadata={"framework": "pytorch", "architecture": "transformer"}
    )
    artifacts.append(checkpoint_artifact)
    
    # Feature definitions
    features_artifact = ReproducibilityArtifact(
        artifact_type=ArtifactType.FEATURE_DEFINITIONS,
        name=f"{model_id}_features",
        version=model_version,
        file_path=f"/models/{model_id}/features_{model_version}.json",
        file_hash=hashlib.sha256(f"{model_id}_{model_version}_features".encode()).hexdigest(),
        file_size_bytes=1024 * 10,  # 10KB
        metadata={"feature_count": 128, "categorical_features": 15}
    )
    artifacts.append(features_artifact)
    
    # Preprocessing pipeline
    preprocessing_artifact = ReproducibilityArtifact(
        artifact_type=ArtifactType.PREPROCESSING_PIPELINE,
        name=f"{model_id}_preprocessing",
        version=model_version,
        file_path=f"/models/{model_id}/preprocessing_{model_version}.pkl",
        file_hash=hashlib.sha256(f"{model_id}_{model_version}_preprocessing".encode()).hexdigest(),
        file_size_bytes=1024 * 5,  # 5KB
        metadata={"scaler_type": "StandardScaler", "encoder_type": "OneHotEncoder"},
        dependencies=[checkpoint_artifact.artifact_id]
    )
    artifacts.append(preprocessing_artifact)
    
    return artifacts

async def capture_model_configuration(model_id: str) -> Dict[str, Any]:
    """Capture model configuration"""
    return {
        "model_type": "transformer",
        "architecture": "bert-base",
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_sequence_length": 512,
        "vocab_size": 30522,
        "dropout_rate": 0.1,
        "activation": "gelu"
    }

async def capture_hyperparameters(model_id: str) -> Dict[str, Any]:
    """Capture model hyperparameters"""
    return {
        "learning_rate": 2e-5,
        "batch_size": 32,
        "num_epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1
    }

def calculate_package_hash(package: ReproducibilityPackage) -> str:
    """Calculate hash of entire reproducibility package"""
    package_content = {
        "model_id": package.model_id,
        "model_version": package.model_version,
        "data_hashes": [ds.data_hash for ds in package.data_snapshots],
        "environment_hash": package.environment_spec.environment_hash,
        "random_seeds": {
            "numpy": package.random_seeds.numpy_seed,
            "python": package.random_seeds.python_seed,
            "torch": package.random_seeds.torch_seed
        },
        "artifact_hashes": [art.file_hash for art in package.artifacts],
        "configuration": package.configuration,
        "hyperparameters": package.hyperparameters
    }
    
    content_json = json.dumps(package_content, sort_keys=True)
    return hashlib.sha256(content_json.encode()).hexdigest()

@app.post("/reproducibility/reproduce")
async def reproduce_model(request: ReproductionRequest):
    """
    Reproduce a model using a reproducibility package
    """
    if request.package_id not in reproducibility_packages:
        raise HTTPException(status_code=404, detail="Reproducibility package not found")
    
    package = reproducibility_packages[request.package_id]
    
    try:
        # Verify package integrity
        current_hash = calculate_package_hash(package)
        if current_hash != package.package_hash:
            raise HTTPException(status_code=400, detail="Package integrity check failed")
        
        # Simulate reproduction process
        reproduction_results = {
            "reproduction_id": str(uuid.uuid4()),
            "package_id": request.package_id,
            "started_at": datetime.utcnow(),
            "environment_setup": "success",
            "data_restoration": "success",
            "seed_initialization": "success",
            "model_training": "success" if request.run_full_validation else "skipped",
            "validation_results": {}
        }
        
        if request.run_full_validation:
            # Simulate validation metrics comparison
            original_metrics = package.validation_metrics
            reproduced_metrics = {
                "accuracy": 0.8501 if "accuracy" in original_metrics else 0.85,
                "f1_score": 0.8234 if "f1_score" in original_metrics else 0.82,
                "precision": 0.8456 if "precision" in original_metrics else 0.84
            }
            
            # Check if reproduction is within tolerance
            validation_passed = True
            metric_differences = {}
            
            for metric, original_value in original_metrics.items():
                if metric in reproduced_metrics:
                    reproduced_value = reproduced_metrics[metric]
                    difference = abs(original_value - reproduced_value)
                    metric_differences[metric] = {
                        "original": original_value,
                        "reproduced": reproduced_value,
                        "difference": difference,
                        "within_tolerance": difference <= package.reproduction_tolerance
                    }
                    
                    if difference > package.reproduction_tolerance:
                        validation_passed = False
            
            reproduction_results["validation_results"] = {
                "validation_passed": validation_passed,
                "metric_differences": metric_differences,
                "tolerance": package.reproduction_tolerance
            }
        
        reproduction_results["completed_at"] = datetime.utcnow()
        reproduction_results["status"] = "success"
        
        # Update package verification results
        package.verification_results = reproduction_results
        if request.run_full_validation and reproduction_results["validation_results"]["validation_passed"]:
            package.status = ReproducibilityStatus.VERIFIED
        
        logger.info(f"Model reproduction completed: {reproduction_results['reproduction_id']}")
        
        return reproduction_results
        
    except Exception as e:
        logger.error(f"Model reproduction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reproduction failed: {e}")

@app.get("/reproducibility/package/{package_id}", response_model=ReproducibilityPackage)
async def get_reproducibility_package(package_id: str):
    """Get reproducibility package details"""
    if package_id not in reproducibility_packages:
        raise HTTPException(status_code=404, detail="Reproducibility package not found")
    
    return reproducibility_packages[package_id]

@app.get("/reproducibility/packages")
async def list_reproducibility_packages(
    tenant_id: Optional[str] = None,
    model_id: Optional[str] = None,
    status: Optional[ReproducibilityStatus] = None,
    limit: int = 50
):
    """List reproducibility packages with filtering"""
    filtered_packages = list(reproducibility_packages.values())
    
    if tenant_id:
        filtered_packages = [p for p in filtered_packages if p.tenant_id == tenant_id]
    
    if model_id:
        filtered_packages = [p for p in filtered_packages if p.model_id == model_id]
    
    if status:
        filtered_packages = [p for p in filtered_packages if p.status == status]
    
    # Sort by creation time, most recent first
    filtered_packages.sort(key=lambda x: x.created_at, reverse=True)
    
    return {
        "packages": filtered_packages[:limit],
        "total_count": len(filtered_packages)
    }

@app.post("/reproducibility/export/{package_id}")
async def export_reproducibility_package(package_id: str):
    """Export reproducibility package as downloadable archive"""
    if package_id not in reproducibility_packages:
        raise HTTPException(status_code=404, detail="Reproducibility package not found")
    
    package = reproducibility_packages[package_id]
    
    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add package metadata
        zip_file.writestr("package_metadata.json", json.dumps(package.dict(), indent=2, default=str))
        
        # Add environment specification
        if package.environment_spec.dockerfile_content:
            zip_file.writestr("Dockerfile", package.environment_spec.dockerfile_content)
        
        if package.environment_spec.conda_yaml:
            zip_file.writestr("environment.yml", package.environment_spec.conda_yaml)
        
        if package.environment_spec.requirements_txt:
            zip_file.writestr("requirements.txt", package.environment_spec.requirements_txt)
        
        # Add configuration files
        zip_file.writestr("model_config.json", json.dumps(package.configuration, indent=2))
        zip_file.writestr("hyperparameters.json", json.dumps(package.hyperparameters, indent=2))
        
        # Add random seeds
        zip_file.writestr("random_seeds.json", json.dumps(package.random_seeds.dict(), indent=2))
        
        # Add data snapshot info (actual data would be referenced)
        data_info = {
            "snapshots": [ds.dict() for ds in package.data_snapshots],
            "note": "Actual data files are stored separately and referenced by storage_path"
        }
        zip_file.writestr("data_snapshots.json", json.dumps(data_info, indent=2, default=str))
        
        # Add artifact info
        artifact_info = {
            "artifacts": [art.dict() for art in package.artifacts],
            "note": "Actual artifact files are stored separately and referenced by file_path"
        }
        zip_file.writestr("artifacts.json", json.dumps(artifact_info, indent=2, default=str))
        
        # Add reproduction script
        reproduction_script = f"""#!/bin/bash
# Reproducibility script for model {package.model_id} version {package.model_version}

echo "Setting up reproducible environment..."

# Set random seeds
export PYTHONHASHSEED={package.random_seeds.python_seed}

# Set environment variables
{chr(10).join([f'export {k}="{v}"' for k, v in package.environment_spec.environment_variables.items()])}

echo "Environment setup complete. Ready for model reproduction."
"""
        zip_file.writestr("reproduce.sh", reproduction_script)
    
    zip_buffer.seek(0)
    
    return {
        "package_id": package_id,
        "export_size_bytes": len(zip_buffer.getvalue()),
        "download_url": f"/reproducibility/download/{package_id}",
        "export_created_at": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RBIA Roll-Forward Reproducibility Service",
        "task": "3.2.32",
        "total_packages": len(reproducibility_packages),
        "artifact_types_supported": [at.value for at in ArtifactType],
        "environment_types_supported": [et.value for et in EnvironmentType]
    }

