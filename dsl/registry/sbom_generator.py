"""
Task 7.3-T12: Generate SBOM for workflows & extensions
Software Bill of Materials generation for transparency and supply chain security
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re

import yaml
import pkg_resources
from packaging import version as pkg_version


class SBOMFormat(Enum):
    """Supported SBOM formats"""
    CYCLONE_DX = "CycloneDX"
    SPDX = "SPDX"
    CUSTOM = "custom"


class ComponentType(Enum):
    """Types of components in SBOM"""
    APPLICATION = "application"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    CONTAINER = "container"
    OPERATING_SYSTEM = "operating-system"
    DEVICE = "device"
    FIRMWARE = "firmware"
    FILE = "file"


class LicenseType(Enum):
    """License types"""
    DECLARED = "declared"
    CONCLUDED = "concluded"
    EXPRESSION = "expression"


@dataclass
class License:
    """License information"""
    license_id: Optional[str] = None  # SPDX license ID
    license_name: Optional[str] = None
    license_text: Optional[str] = None
    license_url: Optional[str] = None
    license_type: LicenseType = LicenseType.DECLARED


@dataclass
class ExternalReference:
    """External reference for components"""
    reference_type: str  # website, distribution, vcs, etc.
    url: str
    comment: Optional[str] = None


@dataclass
class Hash:
    """Hash information for components"""
    algorithm: str  # SHA-1, SHA-256, SHA-512, MD5
    value: str


@dataclass
class Component:
    """SBOM component"""
    component_id: str
    name: str
    version: str
    component_type: ComponentType
    supplier: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    group: Optional[str] = None
    description: Optional[str] = None
    scope: str = "required"  # required, optional, excluded
    hashes: List[Hash] = field(default_factory=list)
    licenses: List[License] = field(default_factory=list)
    copyright: Optional[str] = None
    cpe: Optional[str] = None  # Common Platform Enumeration
    purl: Optional[str] = None  # Package URL
    external_references: List[ExternalReference] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert component to dictionary"""
        result = {
            "bom-ref": self.component_id,
            "type": self.component_type.value,
            "name": self.name,
            "version": self.version,
            "scope": self.scope
        }
        
        if self.supplier:
            result["supplier"] = {"name": self.supplier}
        if self.author:
            result["author"] = self.author
        if self.publisher:
            result["publisher"] = {"name": self.publisher}
        if self.group:
            result["group"] = self.group
        if self.description:
            result["description"] = self.description
        if self.copyright:
            result["copyright"] = self.copyright
        if self.cpe:
            result["cpe"] = self.cpe
        if self.purl:
            result["purl"] = self.purl
        
        if self.hashes:
            result["hashes"] = [
                {"alg": h.algorithm, "content": h.value}
                for h in self.hashes
            ]
        
        if self.licenses:
            result["licenses"] = []
            for license_info in self.licenses:
                license_dict = {}
                if license_info.license_id:
                    license_dict["license"] = {"id": license_info.license_id}
                elif license_info.license_name:
                    license_dict["license"] = {"name": license_info.license_name}
                if license_info.license_text:
                    license_dict["license"]["text"] = {
                        "content": license_info.license_text
                    }
                if license_info.license_url:
                    license_dict["license"]["url"] = license_info.license_url
                result["licenses"].append(license_dict)
        
        if self.external_references:
            result["externalReferences"] = [
                {
                    "type": ref.reference_type,
                    "url": ref.url,
                    **({"comment": ref.comment} if ref.comment else {})
                }
                for ref in self.external_references
            ]
        
        if self.properties:
            result["properties"] = [
                {"name": k, "value": v}
                for k, v in self.properties.items()
            ]
        
        return result


@dataclass
class Dependency:
    """Dependency relationship between components"""
    ref: str  # Reference to component
    depends_on: List[str] = field(default_factory=list)  # List of component refs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dependency to dictionary"""
        return {
            "ref": self.ref,
            "dependsOn": self.depends_on
        }


@dataclass
class Vulnerability:
    """Vulnerability information"""
    vulnerability_id: str
    source: str  # NVD, GitHub, etc.
    references: List[ExternalReference] = field(default_factory=list)
    ratings: List[Dict[str, Any]] = field(default_factory=list)
    cwes: List[int] = field(default_factory=list)
    description: Optional[str] = None
    detail: Optional[str] = None
    recommendation: Optional[str] = None
    advisories: List[ExternalReference] = field(default_factory=list)
    created: Optional[datetime] = None
    published: Optional[datetime] = None
    updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vulnerability to dictionary"""
        result = {
            "id": self.vulnerability_id,
            "source": {"name": self.source}
        }
        
        if self.references:
            result["references"] = [ref.__dict__ for ref in self.references]
        if self.ratings:
            result["ratings"] = self.ratings
        if self.cwes:
            result["cwes"] = [{"id": cwe} for cwe in self.cwes]
        if self.description:
            result["description"] = self.description
        if self.detail:
            result["detail"] = self.detail
        if self.recommendation:
            result["recommendation"] = self.recommendation
        if self.advisories:
            result["advisories"] = [adv.__dict__ for adv in self.advisories]
        if self.created:
            result["created"] = self.created.isoformat()
        if self.published:
            result["published"] = self.published.isoformat()
        if self.updated:
            result["updated"] = self.updated.isoformat()
        
        return result


@dataclass
class SBOM:
    """Software Bill of Materials"""
    bom_format: SBOMFormat
    spec_version: str
    serial_number: str
    version: int
    metadata: Dict[str, Any]
    components: List[Component] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SBOM to dictionary"""
        result = {
            "bomFormat": self.bom_format.value,
            "specVersion": self.spec_version,
            "serialNumber": self.serial_number,
            "version": self.version,
            "metadata": self.metadata
        }
        
        if self.components:
            result["components"] = [comp.to_dict() for comp in self.components]
        
        if self.dependencies:
            result["dependencies"] = [dep.to_dict() for dep in self.dependencies]
        
        if self.vulnerabilities:
            result["vulnerabilities"] = [vuln.to_dict() for vuln in self.vulnerabilities]
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert SBOM to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def get_component_count(self) -> int:
        """Get total number of components"""
        return len(self.components)
    
    def get_dependency_count(self) -> int:
        """Get total number of dependencies"""
        return sum(len(dep.depends_on) for dep in self.dependencies)
    
    def get_license_count(self) -> int:
        """Get number of unique licenses"""
        licenses = set()
        for component in self.components:
            for license_info in component.licenses:
                if license_info.license_id:
                    licenses.add(license_info.license_id)
                elif license_info.license_name:
                    licenses.add(license_info.license_name)
        return len(licenses)
    
    def get_vulnerability_count(self) -> int:
        """Get total number of vulnerabilities"""
        return len(self.vulnerabilities)


class SBOMGenerator:
    """
    Generator for Software Bill of Materials (SBOM) for workflows and extensions
    """
    
    def __init__(self):
        self.known_licenses = self._load_known_licenses()
        self.vulnerability_db = {}  # Would integrate with vulnerability databases
    
    def generate_workflow_sbom(
        self,
        workflow_id: str,
        workflow_name: str,
        workflow_version: str,
        dsl_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SBOM:
        """
        Generate SBOM for a workflow
        
        Args:
            workflow_id: Unique workflow identifier
            workflow_name: Human-readable workflow name
            workflow_version: Workflow version
            dsl_content: DSL content to analyze
            metadata: Additional metadata
            
        Returns:
            Generated SBOM
        """
        # Parse DSL to extract dependencies
        dependencies = self._parse_dsl_dependencies(dsl_content)
        
        # Create main workflow component
        main_component = Component(
            component_id=workflow_id,
            name=workflow_name,
            version=workflow_version,
            component_type=ComponentType.APPLICATION,
            description=f"RBA Workflow: {workflow_name}",
            scope="required"
        )
        
        # Add hash of DSL content
        dsl_hash = hashlib.sha256(dsl_content.encode()).hexdigest()
        main_component.hashes.append(Hash(algorithm="SHA-256", value=dsl_hash))
        
        # Create SBOM metadata
        sbom_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": [
                {
                    "vendor": "RevAI Pro",
                    "name": "SBOM Generator",
                    "version": "1.0.0"
                }
            ],
            "component": main_component.to_dict(),
            "properties": [
                {"name": "workflow_type", "value": "rba"},
                {"name": "dsl_version", "value": "1.0.0"}
            ]
        }
        
        if metadata:
            sbom_metadata.update(metadata)
        
        # Create SBOM
        sbom = SBOM(
            bom_format=SBOMFormat.CYCLONE_DX,
            spec_version="1.4",
            serial_number=f"urn:uuid:{uuid.uuid4()}",
            version=1,
            metadata=sbom_metadata,
            components=[main_component]
        )
        
        # Add dependency components
        for dep in dependencies:
            component = self._create_dependency_component(dep)
            sbom.components.append(component)
        
        # Create dependency relationships
        if dependencies:
            main_dependency = Dependency(
                ref=workflow_id,
                depends_on=[dep["component_id"] for dep in dependencies]
            )
            sbom.dependencies.append(main_dependency)
        
        # Add Python runtime dependencies
        python_deps = self._get_python_dependencies()
        for dep in python_deps:
            component = self._create_python_component(dep)
            sbom.components.append(component)
        
        # Scan for vulnerabilities
        vulnerabilities = self._scan_vulnerabilities(sbom.components)
        sbom.vulnerabilities.extend(vulnerabilities)
        
        return sbom
    
    def generate_extension_sbom(
        self,
        extension_id: str,
        extension_name: str,
        extension_version: str,
        extension_path: Path
    ) -> SBOM:
        """
        Generate SBOM for a workflow extension
        
        Args:
            extension_id: Unique extension identifier
            extension_name: Human-readable extension name
            extension_version: Extension version
            extension_path: Path to extension files
            
        Returns:
            Generated SBOM
        """
        # Analyze extension files
        files = list(extension_path.rglob("*"))
        
        # Create main extension component
        main_component = Component(
            component_id=extension_id,
            name=extension_name,
            version=extension_version,
            component_type=ComponentType.LIBRARY,
            description=f"RBA Extension: {extension_name}",
            scope="required"
        )
        
        # Calculate hash of all files
        file_hashes = []
        for file_path in files:
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    file_hashes.append(file_hash)
        
        # Combined hash
        combined_hash = hashlib.sha256(''.join(sorted(file_hashes)).encode()).hexdigest()
        main_component.hashes.append(Hash(algorithm="SHA-256", value=combined_hash))
        
        # Create SBOM
        sbom = SBOM(
            bom_format=SBOMFormat.CYCLONE_DX,
            spec_version="1.4",
            serial_number=f"urn:uuid:{uuid.uuid4()}",
            version=1,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tools": [
                    {
                        "vendor": "RevAI Pro",
                        "name": "SBOM Generator",
                        "version": "1.0.0"
                    }
                ],
                "component": main_component.to_dict()
            },
            components=[main_component]
        )
        
        # Add file components
        for file_path in files:
            if file_path.is_file() and file_path.suffix in ['.py', '.yaml', '.yml', '.json']:
                file_component = self._create_file_component(file_path, extension_path)
                sbom.components.append(file_component)
        
        return sbom
    
    def _parse_dsl_dependencies(self, dsl_content: str) -> List[Dict[str, Any]]:
        """Parse DSL content to extract dependencies"""
        dependencies = []
        
        try:
            # Parse YAML DSL
            dsl_data = yaml.safe_load(dsl_content)
            
            # Extract connector dependencies
            if isinstance(dsl_data, dict):
                self._extract_connectors(dsl_data, dependencies)
                self._extract_ml_models(dsl_data, dependencies)
                self._extract_external_apis(dsl_data, dependencies)
        
        except yaml.YAMLError:
            # If YAML parsing fails, try to extract dependencies with regex
            dependencies.extend(self._extract_dependencies_regex(dsl_content))
        
        return dependencies
    
    def _extract_connectors(self, dsl_data: Dict[str, Any], dependencies: List[Dict[str, Any]]) -> None:
        """Extract connector dependencies from DSL"""
        # Look for connector references in workflow steps
        if "workflow" in dsl_data and "steps" in dsl_data["workflow"]:
            for step in dsl_data["workflow"]["steps"]:
                if isinstance(step, dict):
                    step_type = step.get("type", "")
                    
                    # Check for connector types
                    if step_type in ["query", "api_call", "webhook"]:
                        connector_name = step.get("connector", "")
                        if connector_name:
                            dependencies.append({
                                "component_id": f"connector-{connector_name}",
                                "name": f"{connector_name} Connector",
                                "version": step.get("connector_version", "latest"),
                                "type": "connector",
                                "source": step_type
                            })
    
    def _extract_ml_models(self, dsl_data: Dict[str, Any], dependencies: List[Dict[str, Any]]) -> None:
        """Extract ML model dependencies from DSL"""
        if "workflow" in dsl_data and "steps" in dsl_data["workflow"]:
            for step in dsl_data["workflow"]["steps"]:
                if isinstance(step, dict) and step.get("type") == "ml_decision":
                    model_name = step.get("model", "")
                    if model_name:
                        dependencies.append({
                            "component_id": f"ml-model-{model_name}",
                            "name": f"ML Model: {model_name}",
                            "version": step.get("model_version", "latest"),
                            "type": "ml_model",
                            "source": "ml_decision"
                        })
    
    def _extract_external_apis(self, dsl_data: Dict[str, Any], dependencies: List[Dict[str, Any]]) -> None:
        """Extract external API dependencies from DSL"""
        if "workflow" in dsl_data and "steps" in dsl_data["workflow"]:
            for step in dsl_data["workflow"]["steps"]:
                if isinstance(step, dict) and step.get("type") == "api_call":
                    api_url = step.get("url", "")
                    if api_url:
                        # Extract domain from URL
                        import urllib.parse
                        parsed_url = urllib.parse.urlparse(api_url)
                        domain = parsed_url.netloc
                        
                        dependencies.append({
                            "component_id": f"external-api-{domain}",
                            "name": f"External API: {domain}",
                            "version": "unknown",
                            "type": "external_api",
                            "source": "api_call",
                            "url": api_url
                        })
    
    def _extract_dependencies_regex(self, dsl_content: str) -> List[Dict[str, Any]]:
        """Extract dependencies using regex patterns"""
        dependencies = []
        
        # Pattern for connector references
        connector_pattern = r'connector:\s*["\']?([^"\'\s]+)["\']?'
        connectors = re.findall(connector_pattern, dsl_content)
        
        for connector in connectors:
            dependencies.append({
                "component_id": f"connector-{connector}",
                "name": f"{connector} Connector",
                "version": "unknown",
                "type": "connector",
                "source": "regex_extraction"
            })
        
        return dependencies
    
    def _create_dependency_component(self, dep: Dict[str, Any]) -> Component:
        """Create a component from dependency information"""
        component = Component(
            component_id=dep["component_id"],
            name=dep["name"],
            version=dep["version"],
            component_type=ComponentType.LIBRARY,
            description=f"Workflow dependency: {dep['name']}",
            scope="required"
        )
        
        # Add properties based on dependency type
        component.properties["dependency_type"] = dep["type"]
        component.properties["source"] = dep["source"]
        
        if "url" in dep:
            component.external_references.append(
                ExternalReference(
                    reference_type="website",
                    url=dep["url"]
                )
            )
        
        return component
    
    def _get_python_dependencies(self) -> List[Dict[str, str]]:
        """Get Python package dependencies"""
        dependencies = []
        
        try:
            # Get installed packages
            installed_packages = [d for d in pkg_resources.working_set]
            
            # Filter for relevant packages
            relevant_packages = [
                "fastapi", "pydantic", "sqlalchemy", "asyncpg", "redis",
                "azure-storage-blob", "azure-identity", "pyjwt", "cryptography"
            ]
            
            for package in installed_packages:
                if package.project_name.lower() in relevant_packages:
                    dependencies.append({
                        "name": package.project_name,
                        "version": package.version,
                        "location": package.location
                    })
        
        except Exception:
            # Fallback to known dependencies
            dependencies = [
                {"name": "fastapi", "version": "0.104.1", "location": ""},
                {"name": "pydantic", "version": "2.5.0", "location": ""},
                {"name": "sqlalchemy", "version": "2.0.23", "location": ""}
            ]
        
        return dependencies
    
    def _create_python_component(self, dep: Dict[str, str]) -> Component:
        """Create a component for Python dependency"""
        component = Component(
            component_id=f"python-{dep['name']}",
            name=dep["name"],
            version=dep["version"],
            component_type=ComponentType.LIBRARY,
            description=f"Python package: {dep['name']}",
            scope="required",
            purl=f"pkg:pypi/{dep['name']}@{dep['version']}"
        )
        
        # Add PyPI reference
        component.external_references.append(
            ExternalReference(
                reference_type="distribution",
                url=f"https://pypi.org/project/{dep['name']}/{dep['version']}/"
            )
        )
        
        # Try to get license information
        license_info = self._get_package_license(dep["name"])
        if license_info:
            component.licenses.append(license_info)
        
        return component
    
    def _create_file_component(self, file_path: Path, base_path: Path) -> Component:
        """Create a component for a file"""
        relative_path = file_path.relative_to(base_path)
        
        with open(file_path, 'rb') as f:
            content = f.read()
            file_hash = hashlib.sha256(content).hexdigest()
        
        component = Component(
            component_id=f"file-{str(relative_path).replace('/', '-')}",
            name=str(relative_path),
            version="1.0.0",
            component_type=ComponentType.FILE,
            description=f"Extension file: {relative_path}",
            scope="required"
        )
        
        component.hashes.append(Hash(algorithm="SHA-256", value=file_hash))
        component.properties["file_path"] = str(relative_path)
        component.properties["file_size"] = str(len(content))
        
        return component
    
    def _get_package_license(self, package_name: str) -> Optional[License]:
        """Get license information for a Python package"""
        try:
            # Try to get license from package metadata
            dist = pkg_resources.get_distribution(package_name)
            if dist.has_metadata('METADATA'):
                metadata = dist.get_metadata('METADATA')
                # Parse metadata for license information
                for line in metadata.split('\n'):
                    if line.startswith('License:'):
                        license_name = line.split(':', 1)[1].strip()
                        return License(
                            license_name=license_name,
                            license_type=LicenseType.DECLARED
                        )
        except Exception:
            pass
        
        # Fallback to known licenses
        known_licenses = {
            "fastapi": "MIT",
            "pydantic": "MIT",
            "sqlalchemy": "MIT",
            "asyncpg": "Apache-2.0",
            "redis": "MIT"
        }
        
        if package_name.lower() in known_licenses:
            return License(
                license_id=known_licenses[package_name.lower()],
                license_type=LicenseType.DECLARED
            )
        
        return None
    
    def _scan_vulnerabilities(self, components: List[Component]) -> List[Vulnerability]:
        """Scan components for known vulnerabilities"""
        vulnerabilities = []
        
        # This would integrate with vulnerability databases like NVD, GitHub Security Advisories, etc.
        # For now, return empty list as placeholder
        
        # Example vulnerability check
        for component in components:
            if component.name == "fastapi" and component.version:
                try:
                    version_obj = pkg_version.parse(component.version)
                    # Example: FastAPI versions before 0.100.0 had a vulnerability
                    if version_obj < pkg_version.parse("0.100.0"):
                        vuln = Vulnerability(
                            vulnerability_id="CVE-2023-EXAMPLE",
                            source="NVD",
                            description="Example vulnerability in FastAPI",
                            recommendation="Upgrade to FastAPI >= 0.100.0"
                        )
                        vulnerabilities.append(vuln)
                except Exception:
                    pass
        
        return vulnerabilities
    
    def _load_known_licenses(self) -> Dict[str, License]:
        """Load known license information"""
        return {
            "MIT": License(
                license_id="MIT",
                license_name="MIT License",
                license_url="https://opensource.org/licenses/MIT"
            ),
            "Apache-2.0": License(
                license_id="Apache-2.0",
                license_name="Apache License 2.0",
                license_url="https://www.apache.org/licenses/LICENSE-2.0"
            ),
            "BSD-3-Clause": License(
                license_id="BSD-3-Clause",
                license_name="BSD 3-Clause License",
                license_url="https://opensource.org/licenses/BSD-3-Clause"
            )
        }


# Example usage
def example_sbom_generation():
    """Example of SBOM generation"""
    generator = SBOMGenerator()
    
    # Example DSL content
    dsl_content = """
    workflow:
      name: "pipeline_hygiene"
      version: "1.0.0"
      steps:
        - name: "fetch_opportunities"
          type: "query"
          connector: "salesforce"
          connector_version: "2.1.0"
          params:
            table: "opportunities"
            filters:
              - "stage != 'Closed Won'"
        
        - name: "score_leads"
          type: "ml_decision"
          model: "lead_scoring_v2"
          model_version: "2.3.1"
          params:
            features: ["company_size", "industry", "engagement_score"]
        
        - name: "send_notification"
          type: "api_call"
          url: "https://hooks.slack.com/services/webhook"
          params:
            message: "Pipeline hygiene check completed"
    """
    
    # Generate SBOM
    sbom = generator.generate_workflow_sbom(
        workflow_id="workflow-123",
        workflow_name="Pipeline Hygiene",
        workflow_version="1.0.0",
        dsl_content=dsl_content,
        metadata={
            "author": "RevOps Team",
            "description": "Automated pipeline hygiene workflow"
        }
    )
    
    # Print SBOM statistics
    print(f"Generated SBOM with:")
    print(f"  - {sbom.get_component_count()} components")
    print(f"  - {sbom.get_dependency_count()} dependencies")
    print(f"  - {sbom.get_license_count()} unique licenses")
    print(f"  - {sbom.get_vulnerability_count()} vulnerabilities")
    
    # Print SBOM JSON (first 1000 characters)
    sbom_json = sbom.to_json()
    print(f"\nSBOM JSON (truncated):\n{sbom_json[:1000]}...")
    
    return sbom


if __name__ == "__main__":
    example_sbom_generation()
