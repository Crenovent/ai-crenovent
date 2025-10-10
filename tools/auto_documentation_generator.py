"""
Auto-Documentation Generator - Task 6.1.40
==========================================
Generate documentation from RBIA plans and workflows
"""

import yaml
import json
from typing import Dict, List, Any
from datetime import datetime
import os


class AutoDocumentationGenerator:
    """Generate documentation from RBIA workflows"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load documentation templates"""
        return {
            'workflow': """
# {name}

**ID**: `{id}`  
**Type**: {type}  
**Version**: {version}  
**Created**: {created_at}

## Description
{description}

## Steps
{steps}

## Governance
{governance}

## ML Models
{models}
""",
            'step': """
### {index}. {name} (`{id}`)
- **Type**: `{type}`
- **Description**: {description}
{params}
"""
        }
    
    def generate_workflow_documentation(self, workflow: Dict[str, Any]) -> str:
        """Generate markdown documentation for a workflow"""
        
        # Extract workflow info
        name = workflow.get('name', 'Unnamed Workflow')
        workflow_id = workflow.get('id', 'unknown')
        workflow_type = workflow.get('type', 'RBA')
        version = workflow.get('version', '1.0.0')
        description = workflow.get('description', 'No description provided')
        created_at = workflow.get('created_at', datetime.utcnow().isoformat())
        
        # Generate steps documentation
        steps_doc = self._generate_steps_doc(workflow.get('steps', []))
        
        # Generate governance documentation
        governance_doc = self._generate_governance_doc(workflow.get('governance', {}))
        
        # Extract ML models
        models_doc = self._extract_models_doc(workflow)
        
        # Fill template
        doc = self.templates['workflow'].format(
            name=name,
            id=workflow_id,
            type=workflow_type,
            version=version,
            created_at=created_at,
            description=description,
            steps=steps_doc,
            governance=governance_doc,
            models=models_doc
        )
        
        return doc
    
    def _generate_steps_doc(self, steps: List[Dict[str, Any]]) -> str:
        """Generate documentation for workflow steps"""
        if not steps:
            return "_No steps defined_"
        
        steps_doc = []
        for idx, step in enumerate(steps, 1):
            step_doc = self.templates['step'].format(
                index=idx,
                name=step.get('name', step.get('id', 'Unnamed Step')),
                id=step.get('id', 'unknown'),
                type=step.get('type', 'unknown'),
                description=step.get('description', ''),
                params=self._format_params(step.get('params', {}))
            )
            steps_doc.append(step_doc)
        
        return '\n'.join(steps_doc)
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format step parameters"""
        if not params:
            return ""
        
        param_lines = []
        for key, value in params.items():
            param_lines.append(f"- **{key}**: `{value}`")
        
        return '\n' + '\n'.join(param_lines)
    
    def _generate_governance_doc(self, governance: Dict[str, Any]) -> str:
        """Generate governance documentation"""
        if not governance:
            return "_No governance configuration_"
        
        doc_lines = []
        for key, value in governance.items():
            doc_lines.append(f"- **{key}**: `{value}`")
        
        return '\n'.join(doc_lines)
    
    def _extract_models_doc(self, workflow: Dict[str, Any]) -> str:
        """Extract ML models used in workflow"""
        models = set()
        
        for step in workflow.get('steps', []):
            if step.get('type', '').startswith('ml_'):
                model_id = step.get('params', {}).get('model_id', step.get('model_id'))
                if model_id:
                    models.add(model_id)
        
        if not models:
            return "_No ML models used_"
        
        model_lines = [f"- `{model}`" for model in sorted(models)]
        return '\n'.join(model_lines)
    
    def generate_swagger_spec(self, api_endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate OpenAPI/Swagger specification"""
        spec = {
            'openapi': '3.0.0',
            'info': {
                'title': 'RBIA Workflow API',
                'version': '1.0.0',
                'description': 'Auto-generated API documentation'
            },
            'paths': {}
        }
        
        for endpoint in api_endpoints:
            path = endpoint.get('path', '/')
            method = endpoint.get('method', 'get').lower()
            
            spec['paths'][path] = {
                method: {
                    'summary': endpoint.get('summary', ''),
                    'description': endpoint.get('description', ''),
                    'responses': {
                        '200': {'description': 'Success'}
                    }
                }
            }
        
        return spec


def generate_docs_for_workflows(workflow_dir: str, output_dir: str):
    """Generate documentation for all workflows in a directory"""
    generator = AutoDocumentationGenerator()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(workflow_dir):
        if filename.endswith(('.yaml', '.yml')):
            filepath = os.path.join(workflow_dir, filename)
            
            with open(filepath, 'r') as f:
                workflow = yaml.safe_load(f)
            
            doc = generator.generate_workflow_documentation(workflow)
            
            output_file = os.path.join(output_dir, filename.replace('.yaml', '.md').replace('.yml', '.md'))
            with open(output_file, 'w') as f:
                f.write(doc)
            
            print(f"✅ Generated documentation: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        generate_docs_for_workflows(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python auto_documentation_generator.py <workflow_dir> <output_dir>")

