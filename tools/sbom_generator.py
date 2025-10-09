"""
SBOM Generation Tool - Task 6.1.53
==================================
Generate Software Bill of Materials (SBOM) for RBIA services
"""

import json
import subprocess
from typing import Dict, List, Any
from datetime import datetime
import hashlib


class SBOMGenerator:
    """Generate SBOM in CycloneDX format"""
    
    def __init__(self):
        self.sbom_version = "1.4"
        self.spec_version = "cyclonedx"
    
    def generate_sbom(self, requirements_file: str = "requirements.txt") -> Dict[str, Any]:
        """Generate SBOM from requirements file"""
        
        # Read dependencies
        dependencies = self._parse_requirements(requirements_file)
        
        # Create SBOM
        sbom = {
            'bomFormat': 'CycloneDX',
            'specVersion': self.sbom_version,
            'serialNumber': f'urn:uuid:{self._generate_uuid()}',
            'version': 1,
            'metadata': {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'tools': [
                    {
                        'vendor': 'RBIA',
                        'name': 'SBOM Generator',
                        'version': '1.0.0'
                    }
                ],
                'component': {
                    'type': 'application',
                    'name': 'RBIA Platform',
                    'version': '1.0.0',
                    'description': 'Rule-Based Intelligence Augmentation Platform'
                }
            },
            'components': dependencies
        }
        
        return sbom
    
    def _parse_requirements(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse requirements.txt file"""
        components = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package==version
                        if '==' in line:
                            name, version = line.split('==')
                        elif '>=' in line:
                            name, version = line.split('>=')
                            version = f">={version}"
                        else:
                            name = line
                            version = "latest"
                        
                        component = {
                            'type': 'library',
                            'name': name,
                            'version': version,
                            'purl': f'pkg:pypi/{name}@{version}'
                        }
                        components.append(component)
        except FileNotFoundError:
            print(f"Requirements file not found: {filepath}")
        
        return components
    
    def _generate_uuid(self) -> str:
        """Generate UUID for SBOM"""
        import uuid
        return str(uuid.uuid4())
    
    def save_sbom(self, sbom: Dict[str, Any], output_file: str = "sbom.json"):
        """Save SBOM to file"""
        with open(output_file, 'w') as f:
            json.dump(sbom, f, indent=2)
        
        print(f"✅ SBOM generated: {output_file}")
    
    def validate_sbom(self, sbom_file: str) -> bool:
        """Validate SBOM format"""
        try:
            with open(sbom_file, 'r') as f:
                sbom = json.load(f)
            
            # Check required fields
            required_fields = ['bomFormat', 'specVersion', 'version', 'components']
            for field in required_fields:
                if field not in sbom:
                    print(f"Missing required field: {field}")
                    return False
            
            print(f"✅ SBOM valid: {len(sbom.get('components', []))} components")
            return True
            
        except Exception as e:
            print(f"SBOM validation failed: {e}")
            return False


def generate_sbom_for_project(
    requirements_file: str = "requirements.txt",
    output_file: str = "sbom.json"
):
    """Generate SBOM for the project"""
    generator = SBOMGenerator()
    
    sbom = generator.generate_sbom(requirements_file)
    generator.save_sbom(sbom, output_file)
    generator.validate_sbom(output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        generate_sbom_for_project(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sbom.json")
    else:
        generate_sbom_for_project()

