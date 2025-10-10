"""
RBA to RBIA Migration Toolkit - Task 6.1.67
===========================================
Migrate RBA workflows to RBIA with ML enhancements
"""

import yaml
import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class RBAtoRBIAMigrator:
    """Migrate RBA workflows to RBIA format"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def migrate_workflow(self, rba_workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate RBA workflow to RBIA format
        
        Enhancements:
        - Add ML prediction nodes
        - Add confidence thresholds
        - Add explainability
        - Add governance hooks
        """
        rbia_workflow = rba_workflow.copy()
        
        # Add RBIA metadata
        rbia_workflow['type'] = 'RBIA'
        rbia_workflow['migrated_from'] = 'RBA'
        rbia_workflow['ml_enabled'] = True
        
        # Enhance steps with ML
        if 'steps' in rbia_workflow:
            enhanced_steps = []
            for step in rbia_workflow['steps']:
                # Identify deterministic rules that can be ML-enhanced
                if step.get('type') in ['filter', 'classify', 'score']:
                    # Add ML equivalent
                    ml_step = self._create_ml_step(step)
                    enhanced_steps.append(ml_step)
                    
                    # Keep original as fallback
                    step['is_fallback'] = True
                    enhanced_steps.append(step)
                else:
                    enhanced_steps.append(step)
            
            rbia_workflow['steps'] = enhanced_steps
        
        # Add governance
        rbia_workflow['governance'] = {
            'confidence_threshold': 0.7,
            'fallback_enabled': True,
            'explainability_required': True,
            'evidence_capture': True
        }
        
        self.logger.info(f"Migrated workflow: {rbia_workflow.get('id', 'unknown')}")
        
        return rbia_workflow
    
    def _create_ml_step(self, rba_step: Dict[str, Any]) -> Dict[str, Any]:
        """Create ML-enhanced version of RBA step"""
        step_type = rba_step.get('type')
        
        ml_step = {
            'id': f"{rba_step.get('id', 'step')}_ml",
            'type': f'ml_{step_type}',
            'model_id': f"{step_type}_model_v1",
            'confidence_threshold': 0.7,
            'explainability_enabled': True,
            'fallback_step': rba_step.get('id'),
            'params': rba_step.get('params', {})
        }
        
        return ml_step
    
    def validate_migration(self, original: Dict[str, Any], migrated: Dict[str, Any]) -> List[str]:
        """Validate migration succeeded"""
        warnings = []
        
        # Check ML steps added
        if migrated.get('ml_enabled') != True:
            warnings.append("ML not enabled in migrated workflow")
        
        # Check governance added
        if 'governance' not in migrated:
            warnings.append("Governance configuration missing")
        
        return warnings


def migrate_rba_file(input_path: str, output_path: str):
    """Migrate RBA workflow file to RBIA"""
    migrator = RBAtoRBIAMigrator()
    
    with open(input_path, 'r') as f:
        rba_workflow = yaml.safe_load(f)
    
    rbia_workflow = migrator.migrate_workflow(rba_workflow)
    
    warnings = migrator.validate_migration(rba_workflow, rbia_workflow)
    if warnings:
        print(f"Migration warnings: {warnings}")
    
    with open(output_path, 'w') as f:
        yaml.dump(rbia_workflow, f, default_flow_style=False)
    
    print(f"✅ Migrated {input_path} -> {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        migrate_rba_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python rba_to_rbia_migrator.py <input.yaml> <output.yaml>")

