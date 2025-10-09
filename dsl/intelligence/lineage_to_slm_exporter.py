"""
Lineage to SLM Export Pipeline - Task 6.1.68
============================================
Export execution lineage to SLM training format
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class LineageToSLMExporter:
    """Export lineage data for SLM fine-tuning"""
    
    def __init__(self, kg_store=None):
        self.kg_store = kg_store
        self.logger = logging.getLogger(__name__)
    
    async def export_traces_for_slm(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Export execution traces to SLM training format
        
        Format: [
            {
                "input": "user query or workflow config",
                "output": "execution result",
                "metadata": {"workflow_id": "...", "success": true, ...}
            }
        ]
        """
        # Query KG for execution traces
        traces = await self._fetch_traces(tenant_id, start_date, end_date)
        
        # Convert to SLM training format
        training_data = []
        for trace in traces:
            example = self._convert_trace_to_training_example(trace)
            if example:
                training_data.append(example)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        self.logger.info(f"Exported {len(training_data)} training examples to {output_path}")
        
        return {
            'total_traces': len(traces),
            'training_examples': len(training_data),
            'output_file': output_path
        }
    
    async def _fetch_traces(self, tenant_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch execution traces from KG"""
        if not self.kg_store:
            return []
        
        # Query would be: SELECT * FROM kg_execution_traces WHERE ...
        # For now, return empty
        return []
    
    def _convert_trace_to_training_example(self, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert execution trace to SLM training example"""
        try:
            return {
                'input': json.dumps(trace.get('inputs', {})),
                'output': json.dumps(trace.get('outputs', {})),
                'metadata': {
                    'workflow_id': trace.get('workflow_id'),
                    'success': trace.get('status') == 'completed',
                    'execution_time_ms': trace.get('execution_time_ms'),
                    'trust_score': trace.get('trust_score', 1.0)
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to convert trace: {e}")
            return None


async def export_lineage_for_slm_training(
    tenant_id: str,
    start_date: datetime,
    end_date: datetime,
    output_path: str = "slm_training_data.json"
):
    """Helper function to export lineage"""
    exporter = LineageToSLMExporter()
    return await exporter.export_traces_for_slm(tenant_id, start_date, end_date, output_path)

