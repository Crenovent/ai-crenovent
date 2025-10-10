"""
Task 7.3-T30: Implement License & IP metadata
Legal clarity with SPDX license IDs and intellectual property tracking
"""

import json
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from pathlib import Path

import requests
from packaging import version as pkg_version


class LicenseType(Enum):
    """Types of licenses"""
    OPEN_SOURCE = "open_source"
    PROPRIETARY = "proprietary"
    COMMERCIAL = "commercial"
    DUAL_LICENSE = "dual_license"
    PUBLIC_DOMAIN = "public_domain"
    UNKNOWN = "unknown"


class IPRightType(Enum):
    """Types of intellectual property rights"""
    COPYRIGHT = "copyright"
    PATENT = "patent"
    TRADEMARK = "trademark"
    TRADE_SECRET = "trade_secret"
    DESIGN_RIGHT = "design_right"


class LicenseCompatibility(Enum):
    """License compatibility levels"""
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


@dataclass
class License:
    """License information with SPDX compliance"""
    license_id: str  # SPDX license identifier
    name: str
    license_type: LicenseType
    url: Optional[str] = None
    text: Optional[str] = None
    spdx_id: Optional[str] = None
    osi_approved: bool = False
    fsf_approved: bool = False
    copyleft: bool = False
    commercial_use: bool = True
    distribution: bool = True
    modification: bool = True
    private_use: bool = True
    patent_grant: bool = False
    conditions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert license to dictionary"""
        return {
            "license_id": self.license_id,
            "name": self.name,
            "license_type": self.license_type.value,
            "url": self.url,
            "spdx_id": self.spdx_id,
            "osi_approved": self.osi_approved,
            "fsf_approved": self.fsf_approved,
            "copyleft": self.copyleft,
            "commercial_use": self.commercial_use,
            "distribution": self.distribution,
            "modification": self.modification,
            "private_use": self.private_use,
            "patent_grant": self.patent_grant,
            "conditions": self.conditions,
            "limitations": self.limitations,
            "permissions": self.permissions
        }
    
    def is_compatible_with(self, other: 'License') -> LicenseCompatibility:
        """Check compatibility with another license"""
        # Simplified compatibility check
        if self.license_id == other.license_id:
            return LicenseCompatibility.COMPATIBLE
        
        # GPL compatibility rules
        if self.spdx_id and "GPL" in self.spdx_id:
            if other.copyleft:
                return LicenseCompatibility.COMPATIBLE
            else:
                return LicenseCompatibility.INCOMPATIBLE
        
        # MIT/Apache compatibility
        if self.spdx_id in ["MIT", "Apache-2.0"] and other.spdx_id in ["MIT", "Apache-2.0", "BSD-3-Clause"]:
            return LicenseCompatibility.COMPATIBLE
        
        # Proprietary licenses
        if self.license_type == LicenseType.PROPRIETARY or other.license_type == LicenseType.PROPRIETARY:
            return LicenseCompatibility.CONDITIONAL
        
        return LicenseCompatibility.UNKNOWN


@dataclass
class IPRight:
    """Intellectual property right information"""
    right_id: str
    right_type: IPRightType
    title: str
    owner: str
    description: Optional[str] = None
    registration_number: Optional[str] = None
    registration_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    jurisdiction: str = "US"
    status: str = "active"  # active, expired, pending, abandoned
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert IP right to dictionary"""
        return {
            "right_id": self.right_id,
            "right_type": self.right_type.value,
            "title": self.title,
            "owner": self.owner,
            "description": self.description,
            "registration_number": self.registration_number,
            "registration_date": self.registration_date.isoformat() if self.registration_date else None,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "jurisdiction": self.jurisdiction,
            "status": self.status
        }


@dataclass
class Attribution:
    """Attribution information for third-party components"""
    component_name: str
    version: str
    author: Optional[str] = None
    copyright_notice: Optional[str] = None
    license: Optional[License] = None
    source_url: Optional[str] = None
    modification_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert attribution to dictionary"""
        return {
            "component_name": self.component_name,
            "version": self.version,
            "author": self.author,
            "copyright_notice": self.copyright_notice,
            "license": self.license.to_dict() if self.license else None,
            "source_url": self.source_url,
            "modification_description": self.modification_description
        }


@dataclass
class LicenseIPMetadata:
    """Complete license and IP metadata for a workflow"""
    workflow_id: str
    workflow_name: str
    workflow_version: str
    
    # Primary license
    primary_license: License
    
    # Additional licenses (for dual licensing, etc.)
    additional_licenses: List[License] = field(default_factory=list)
    
    # Copyright information
    copyright_holder: str = ""
    copyright_year: int = field(default_factory=lambda: datetime.now().year)
    copyright_notice: str = ""
    
    # IP rights
    ip_rights: List[IPRight] = field(default_factory=list)
    
    # Third-party attributions
    attributions: List[Attribution] = field(default_factory=list)
    
    # Legal notices
    legal_notices: List[str] = field(default_factory=list)
    disclaimer: Optional[str] = None
    
    # Compliance information
    export_control_classification: Optional[str] = None
    regulatory_compliance: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_all_licenses(self) -> List[License]:
        """Get all licenses (primary + additional)"""
        return [self.primary_license] + self.additional_licenses
    
    def check_license_compatibility(self) -> Dict[str, Any]:
        """Check compatibility between all licenses"""
        licenses = self.get_all_licenses()
        compatibility_matrix = {}
        
        for i, license1 in enumerate(licenses):
            for j, license2 in enumerate(licenses):
                if i != j:
                    key = f"{license1.license_id}-{license2.license_id}"
                    compatibility_matrix[key] = license1.is_compatible_with(license2).value
        
        return {
            "compatible": all(comp != "incompatible" for comp in compatibility_matrix.values()),
            "compatibility_matrix": compatibility_matrix,
            "warnings": [k for k, v in compatibility_matrix.items() if v == "conditional"],
            "conflicts": [k for k, v in compatibility_matrix.items() if v == "incompatible"]
        }
    
    def generate_license_notice(self) -> str:
        """Generate complete license notice"""
        lines = []
        
        # Copyright notice
        if self.copyright_notice:
            lines.append(self.copyright_notice)
        else:
            lines.append(f"Copyright (c) {self.copyright_year} {self.copyright_holder}")
        
        lines.append("")
        
        # Primary license
        lines.append(f"Licensed under the {self.primary_license.name}")
        if self.primary_license.url:
            lines.append(f"License URL: {self.primary_license.url}")
        
        # Additional licenses
        if self.additional_licenses:
            lines.append("")
            lines.append("Additional licenses:")
            for license_info in self.additional_licenses:
                lines.append(f"- {license_info.name}")
                if license_info.url:
                    lines.append(f"  URL: {license_info.url}")
        
        # Third-party attributions
        if self.attributions:
            lines.append("")
            lines.append("Third-party components:")
            for attr in self.attributions:
                lines.append(f"- {attr.component_name} v{attr.version}")
                if attr.author:
                    lines.append(f"  Author: {attr.author}")
                if attr.license:
                    lines.append(f"  License: {attr.license.name}")
                if attr.copyright_notice:
                    lines.append(f"  Copyright: {attr.copyright_notice}")
        
        # Legal notices
        if self.legal_notices:
            lines.append("")
            lines.append("Legal notices:")
            for notice in self.legal_notices:
                lines.append(f"- {notice}")
        
        # Disclaimer
        if self.disclaimer:
            lines.append("")
            lines.append("DISCLAIMER:")
            lines.append(self.disclaimer)
        
        return "\n".join(lines)
    
    def to_spdx_document(self) -> Dict[str, Any]:
        """Generate SPDX document"""
        return {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": f"SPDXRef-DOCUMENT-{self.workflow_id}",
            "name": f"{self.workflow_name}-{self.workflow_version}",
            "documentNamespace": f"https://registry.revai.pro/workflows/{self.workflow_id}",
            "creationInfo": {
                "created": self.created_at.isoformat(),
                "creators": ["Tool: RevAI Pro Registry"],
                "licenseListVersion": "3.21"
            },
            "packages": [
                {
                    "SPDXID": f"SPDXRef-Package-{self.workflow_id}",
                    "name": self.workflow_name,
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                    "licenseConcluded": self.primary_license.spdx_id or "NOASSERTION",
                    "licenseDeclared": self.primary_license.spdx_id or "NOASSERTION",
                    "copyrightText": self.copyright_notice or "NOASSERTION",
                    "versionInfo": self.workflow_version
                }
            ]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "primary_license": self.primary_license.to_dict(),
            "additional_licenses": [lic.to_dict() for lic in self.additional_licenses],
            "copyright_holder": self.copyright_holder,
            "copyright_year": self.copyright_year,
            "copyright_notice": self.copyright_notice,
            "ip_rights": [right.to_dict() for right in self.ip_rights],
            "attributions": [attr.to_dict() for attr in self.attributions],
            "legal_notices": self.legal_notices,
            "disclaimer": self.disclaimer,
            "export_control_classification": self.export_control_classification,
            "regulatory_compliance": self.regulatory_compliance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class SPDXLicenseRegistry:
    """Registry of SPDX licenses with caching"""
    
    def __init__(self):
        self.licenses: Dict[str, License] = {}
        self.license_list_url = "https://raw.githubusercontent.com/spdx/license-list-data/main/json/licenses.json"
        self._load_builtin_licenses()
    
    def _load_builtin_licenses(self) -> None:
        """Load common built-in licenses"""
        builtin_licenses = [
            License(
                license_id="MIT",
                name="MIT License",
                license_type=LicenseType.OPEN_SOURCE,
                spdx_id="MIT",
                url="https://opensource.org/licenses/MIT",
                osi_approved=True,
                commercial_use=True,
                distribution=True,
                modification=True,
                private_use=True,
                conditions=["include-copyright"],
                permissions=["commercial-use", "distribution", "modification", "private-use"]
            ),
            License(
                license_id="Apache-2.0",
                name="Apache License 2.0",
                license_type=LicenseType.OPEN_SOURCE,
                spdx_id="Apache-2.0",
                url="https://www.apache.org/licenses/LICENSE-2.0",
                osi_approved=True,
                patent_grant=True,
                commercial_use=True,
                distribution=True,
                modification=True,
                private_use=True,
                conditions=["include-copyright", "document-changes"],
                permissions=["commercial-use", "distribution", "modification", "patent-use", "private-use"]
            ),
            License(
                license_id="GPL-3.0",
                name="GNU General Public License v3.0",
                license_type=LicenseType.OPEN_SOURCE,
                spdx_id="GPL-3.0-only",
                url="https://www.gnu.org/licenses/gpl-3.0.html",
                osi_approved=True,
                fsf_approved=True,
                copyleft=True,
                commercial_use=True,
                distribution=True,
                modification=True,
                private_use=True,
                conditions=["include-copyright", "document-changes", "disclose-source", "same-license"],
                limitations=["liability", "warranty"]
            ),
            License(
                license_id="BSD-3-Clause",
                name="BSD 3-Clause \"New\" or \"Revised\" License",
                license_type=LicenseType.OPEN_SOURCE,
                spdx_id="BSD-3-Clause",
                url="https://opensource.org/licenses/BSD-3-Clause",
                osi_approved=True,
                commercial_use=True,
                distribution=True,
                modification=True,
                private_use=True,
                conditions=["include-copyright"],
                limitations=["liability", "warranty"]
            ),
            License(
                license_id="PROPRIETARY",
                name="Proprietary License",
                license_type=LicenseType.PROPRIETARY,
                commercial_use=False,
                distribution=False,
                modification=False,
                private_use=True,
                limitations=["liability", "warranty", "commercial-use", "distribution", "modification"]
            )
        ]
        
        for license_info in builtin_licenses:
            self.licenses[license_info.license_id] = license_info
    
    def get_license(self, license_id: str) -> Optional[License]:
        """Get license by ID"""
        return self.licenses.get(license_id)
    
    def search_licenses(self, query: str) -> List[License]:
        """Search licenses by name or ID"""
        query_lower = query.lower()
        results = []
        
        for license_info in self.licenses.values():
            if (query_lower in license_info.license_id.lower() or 
                query_lower in license_info.name.lower()):
                results.append(license_info)
        
        return results
    
    def load_spdx_licenses(self) -> None:
        """Load licenses from SPDX license list (requires internet)"""
        try:
            response = requests.get(self.license_list_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            for license_data in data.get("licenses", []):
                license_info = License(
                    license_id=license_data["licenseId"],
                    name=license_data["name"],
                    license_type=LicenseType.OPEN_SOURCE if license_data.get("isOsiApproved", False) else LicenseType.UNKNOWN,
                    spdx_id=license_data["licenseId"],
                    url=license_data.get("reference", ""),
                    osi_approved=license_data.get("isOsiApproved", False),
                    fsf_approved=license_data.get("isFsfLibre", False)
                )
                self.licenses[license_info.license_id] = license_info
                
        except Exception as e:
            print(f"Warning: Could not load SPDX licenses: {e}")
    
    def get_all_licenses(self) -> List[License]:
        """Get all available licenses"""
        return list(self.licenses.values())


class LicenseIPManager:
    """Manager for license and IP metadata"""
    
    def __init__(self):
        self.spdx_registry = SPDXLicenseRegistry()
        self.metadata_store: Dict[str, LicenseIPMetadata] = {}
    
    def create_metadata(
        self,
        workflow_id: str,
        workflow_name: str,
        workflow_version: str,
        primary_license_id: str,
        copyright_holder: str,
        copyright_year: Optional[int] = None
    ) -> LicenseIPMetadata:
        """Create license and IP metadata for a workflow"""
        
        primary_license = self.spdx_registry.get_license(primary_license_id)
        if not primary_license:
            # Create unknown license
            primary_license = License(
                license_id=primary_license_id,
                name=f"Unknown License ({primary_license_id})",
                license_type=LicenseType.UNKNOWN
            )
        
        metadata = LicenseIPMetadata(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            workflow_version=workflow_version,
            primary_license=primary_license,
            copyright_holder=copyright_holder,
            copyright_year=copyright_year or datetime.now().year
        )
        
        self.metadata_store[workflow_id] = metadata
        return metadata
    
    def add_third_party_attribution(
        self,
        workflow_id: str,
        component_name: str,
        version: str,
        license_id: str,
        author: Optional[str] = None,
        copyright_notice: Optional[str] = None,
        source_url: Optional[str] = None
    ) -> None:
        """Add third-party component attribution"""
        
        if workflow_id not in self.metadata_store:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        license_info = self.spdx_registry.get_license(license_id)
        if not license_info:
            license_info = License(
                license_id=license_id,
                name=f"Unknown License ({license_id})",
                license_type=LicenseType.UNKNOWN
            )
        
        attribution = Attribution(
            component_name=component_name,
            version=version,
            author=author,
            copyright_notice=copyright_notice,
            license=license_info,
            source_url=source_url
        )
        
        self.metadata_store[workflow_id].attributions.append(attribution)
        self.metadata_store[workflow_id].updated_at = datetime.now(timezone.utc)
    
    def add_ip_right(
        self,
        workflow_id: str,
        right_type: IPRightType,
        title: str,
        owner: str,
        registration_number: Optional[str] = None,
        jurisdiction: str = "US"
    ) -> None:
        """Add intellectual property right"""
        
        if workflow_id not in self.metadata_store:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        ip_right = IPRight(
            right_id=str(uuid.uuid4()),
            right_type=right_type,
            title=title,
            owner=owner,
            registration_number=registration_number,
            jurisdiction=jurisdiction
        )
        
        self.metadata_store[workflow_id].ip_rights.append(ip_right)
        self.metadata_store[workflow_id].updated_at = datetime.now(timezone.utc)
    
    def validate_license_compliance(self, workflow_id: str) -> Dict[str, Any]:
        """Validate license compliance for a workflow"""
        
        if workflow_id not in self.metadata_store:
            return {"valid": False, "error": f"Workflow {workflow_id} not found"}
        
        metadata = self.metadata_store[workflow_id]
        
        # Check license compatibility
        compatibility = metadata.check_license_compatibility()
        
        # Check required fields
        required_fields = {
            "primary_license": metadata.primary_license is not None,
            "copyright_holder": bool(metadata.copyright_holder),
            "copyright_year": metadata.copyright_year > 0
        }
        
        # Check third-party attributions
        attribution_issues = []
        for attr in metadata.attributions:
            if not attr.license:
                attribution_issues.append(f"Missing license for {attr.component_name}")
            if not attr.copyright_notice and not attr.author:
                attribution_issues.append(f"Missing copyright/author for {attr.component_name}")
        
        return {
            "valid": compatibility["compatible"] and all(required_fields.values()) and not attribution_issues,
            "license_compatibility": compatibility,
            "required_fields": required_fields,
            "attribution_issues": attribution_issues,
            "recommendations": self._generate_compliance_recommendations(metadata, compatibility, attribution_issues)
        }
    
    def _generate_compliance_recommendations(
        self,
        metadata: LicenseIPMetadata,
        compatibility: Dict[str, Any],
        attribution_issues: List[str]
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if not compatibility["compatible"]:
            recommendations.append("Review license compatibility issues before distribution")
        
        if compatibility["warnings"]:
            recommendations.append("Review conditional license compatibility warnings")
        
        if not metadata.copyright_holder:
            recommendations.append("Add copyright holder information")
        
        if attribution_issues:
            recommendations.append("Complete third-party attribution information")
        
        if not metadata.disclaimer:
            recommendations.append("Consider adding a legal disclaimer")
        
        return recommendations
    
    def export_license_report(self, workflow_id: str) -> Dict[str, Any]:
        """Export comprehensive license report"""
        
        if workflow_id not in self.metadata_store:
            return {"error": f"Workflow {workflow_id} not found"}
        
        metadata = self.metadata_store[workflow_id]
        compliance = self.validate_license_compliance(workflow_id)
        
        return {
            "workflow_info": {
                "id": metadata.workflow_id,
                "name": metadata.workflow_name,
                "version": metadata.workflow_version
            },
            "license_metadata": metadata.to_dict(),
            "compliance_status": compliance,
            "license_notice": metadata.generate_license_notice(),
            "spdx_document": metadata.to_spdx_document(),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }


# Example usage
def example_license_ip_metadata():
    """Example of license and IP metadata management"""
    
    # Create manager
    manager = LicenseIPManager()
    
    # Create metadata for a workflow
    metadata = manager.create_metadata(
        workflow_id="workflow-123",
        workflow_name="Pipeline Hygiene",
        workflow_version="1.0.0",
        primary_license_id="MIT",
        copyright_holder="RevAI Pro Inc.",
        copyright_year=2024
    )
    
    # Add third-party attributions
    manager.add_third_party_attribution(
        workflow_id="workflow-123",
        component_name="FastAPI",
        version="0.104.1",
        license_id="MIT",
        author="Sebastián Ramirez",
        copyright_notice="Copyright (c) 2018 Sebastián Ramirez",
        source_url="https://github.com/tiangolo/fastapi"
    )
    
    manager.add_third_party_attribution(
        workflow_id="workflow-123",
        component_name="Pydantic",
        version="2.5.0",
        license_id="MIT",
        author="Samuel Colvin",
        source_url="https://github.com/pydantic/pydantic"
    )
    
    # Add IP right
    manager.add_ip_right(
        workflow_id="workflow-123",
        right_type=IPRightType.COPYRIGHT,
        title="Pipeline Hygiene Workflow Algorithm",
        owner="RevAI Pro Inc.",
        jurisdiction="US"
    )
    
    # Validate compliance
    compliance = manager.validate_license_compliance("workflow-123")
    print(f"License compliance: {compliance['valid']}")
    
    # Generate license notice
    notice = metadata.generate_license_notice()
    print("\n=== LICENSE NOTICE ===")
    print(notice)
    
    # Export full report
    report = manager.export_license_report("workflow-123")
    print(f"\n=== COMPLIANCE REPORT ===")
    print(f"Valid: {report['compliance_status']['valid']}")
    print(f"Recommendations: {report['compliance_status']['recommendations']}")
    
    return metadata, manager


if __name__ == "__main__":
    example_license_ip_metadata()
