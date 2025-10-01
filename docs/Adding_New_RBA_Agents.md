# Adding New RBA Agents

This guide explains how to add new RBA (Rule-Based Automation) agents to the modular architecture.

## **Overview**

The modular RBA architecture uses:
- **Dynamic Discovery**: New agents are automatically discovered without code changes
- **Registry-Based**: Agents register themselves with the `RBAAgentRegistry`
- **Single-Purpose**: Each agent handles ONE specific analysis type
- **Configuration-Driven**: Business rules are externalized to YAML files

## **Step-by-Step Process**

### **Step 1: Create the New RBA Agent File**

Create a new file in `dsl/operators/rba/` following the naming convention: `{analysis_name}_rba_agent.py`

```python
"""
{Analysis Name} RBA Agent
Single-purpose, focused RBA agent for {analysis description} only

This agent ONLY handles {specific functionality}.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_rba_agent import BaseRBAAgent

logger = logging.getLogger(__name__)

class {AnalysisName}RBAAgent(BaseRBAAgent):
    """
    Single-purpose RBA agent for {analysis description}
    
    Features:
    - ONLY handles {specific functionality}
    - Configuration-driven {rules/thresholds}
    - Lightweight and focused
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "{analysis_name}"
    AGENT_DESCRIPTION = "{Brief description of what this agent does}"
    SUPPORTED_ANALYSIS_TYPES = ["{analysis_type1}", "{analysis_type2}"]
    
    @property
    def analysis_type(self) -> str:
        return self.AGENT_NAME
    
    @property
    def description(self) -> str:
        return self.AGENT_DESCRIPTION
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Agent-specific defaults
        self.default_config = {
            'threshold_value': 30,
            'severity_levels': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            'include_analysis': True
        }
    
    async def _validate_agent_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate agent-specific configuration"""
        errors = []
        
        # Add validation logic here
        if config.get('threshold_value', 0) < 0:
            errors.append("threshold_value must be non-negative")
        
        return errors
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 {self.analysis_type.replace('_', ' ').title()} RBA: Analyzing {len(opportunities)} opportunities")
            
            # Main analysis logic
            flagged_deals = []
            
            for opp in opportunities:
                if opp is None:
                    continue  # Skip None opportunities
                
                # Your analysis logic here
                analysis_result = self._analyze_opportunity(opp, config)
                if analysis_result:
                    flagged_deals.append(analysis_result)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': self.analysis_type,
                'total_opportunities': len(opportunities),
                'flagged_opportunities': len(flagged_deals),
                'flagged_deals': flagged_deals,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ {self.analysis_type.replace('_', ' ').title()} RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _analyze_opportunity(self, opp: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single opportunity"""
        
        # Example analysis logic
        threshold = config.get('threshold_value', 30)
        
        # Use safe accessors from base class
        amount = self._safe_float(opp.get('Amount'))
        probability = self._safe_float(opp.get('Probability'))
        
        # Your business logic here
        if amount > threshold:
            return {
                'opportunity_id': opp.get('Id', 'Unknown'),
                'opportunity_name': opp.get('Name', 'Unnamed'),
                'account_name': (opp.get('Account') or {}).get('Name', 'Unknown'),
                'owner_name': (opp.get('Owner') or {}).get('Name', 'Unassigned'),
                'amount': amount,
                'probability': probability,
                'severity': 'HIGH' if amount > threshold * 2 else 'MEDIUM',
                'risk_level': 'HIGH' if amount > threshold * 2 else 'MEDIUM',
                'issue_type': '{ANALYSIS_TYPE}_ISSUE',
                'priority': 'HIGH',
                'description': f"Issue detected: {analysis_description}",
                'recommended_action': "Review and take appropriate action",
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        return None
    
    @classmethod
    def get_agent_metadata(cls) -> Dict[str, Any]:
        """Get agent metadata for registry"""
        return {
            'agent_type': cls.AGENT_TYPE,
            'agent_name': cls.AGENT_NAME,
            'agent_description': cls.AGENT_DESCRIPTION,
            'supported_analysis_types': cls.SUPPORTED_ANALYSIS_TYPES,
            'class_name': cls.__name__,
            'module_path': cls.__module__
        }
```

### **Step 2: Update the RBA Module Init File**

Add your new agent to `dsl/operators/rba/__init__.py`:

```python
# Add to imports
from .{analysis_name}_rba_agent import {AnalysisName}RBAAgent

# Add to __all__ list
__all__ = [
    'BaseRBAAgent',
    # ... existing agents ...
    '{AnalysisName}RBAAgent',  # Add your agent here
]
```

### **Step 3: Update the Registry (Optional)**

The registry automatically discovers agents, but if you want to verify:

```python
# In dsl/registry/rba_agent_registry.py, the agents are auto-discovered
# No manual changes needed - your agent will be automatically registered!
```

### **Step 4: Add Business Rules (Optional)**

If your agent needs configurable business rules, add them to `dsl/rules/business_rules.yaml`:

```yaml
{analysis_name}_rules:
  scoring_factors:
    factor1:
      weight: 25
      thresholds:
        low: 10
        medium: 20
        high: 30
  
  conditions:
    - name: "high_risk_condition"
      field: "Amount"
      operator: "greater_than"
      value: 100000
      score: 30
```

### **Step 5: Test Your Agent**

Create a simple test:

```python
# test_new_agent.py
import asyncio
from dsl.orchestration.dynamic_rba_orchestrator import dynamic_rba_orchestrator

async def test_new_agent():
    # Test data
    opportunities = [
        {
            'Id': 'TEST001',
            'Name': 'Test Opportunity',
            'Amount': 50000,
            'Probability': 75,
            'StageName': 'Proposal',
            'Account': {'Name': 'Test Account'},
            'Owner': {'Name': 'Test Owner'}
        }
    ]
    
    # Execute your new analysis type
    result = await dynamic_rba_orchestrator.execute_rba_analysis(
        analysis_type='{your_analysis_type}',
        opportunities=opportunities,
        config={},
        tenant_id="1300",
        user_id="1319"
    )
    
    print(f"Success: {result['success']}")
    print(f"Flagged: {result.get('flagged_opportunities', 0)}")

if __name__ == "__main__":
    asyncio.run(test_new_agent())
```

## **Key Guidelines**

### **1. Follow Single Responsibility Principle**
- Each agent handles ONE analysis type
- Keep agents lightweight and focused
- Avoid combining multiple analyses in one agent

### **2. Use Safe Data Access**
- Always use `self._safe_float()`, `self._safe_int()`, `self._safe_str()`
- Check for `None` opportunities: `if opp is None: continue`
- Use safe dictionary access: `(opp.get('Account') or {}).get('Name', 'Unknown')`

### **3. Configuration-Driven**
- Externalize business rules to YAML files
- Use `self.default_config` for default values
- Make thresholds and rules configurable

### **4. Proper Error Handling**
- Wrap main logic in try-catch
- Return structured error responses
- Log errors with context

### **5. Consistent Naming**
- File: `{analysis_name}_rba_agent.py`
- Class: `{AnalysisName}RBAAgent`
- Analysis types: lowercase with underscores

## **Example: Adding a "Revenue Risk" Agent**

```python
# File: dsl/operators/rba/revenue_risk_rba_agent.py
class RevenueRiskRBAAgent(BaseRBAAgent):
    AGENT_NAME = "revenue_risk"
    AGENT_DESCRIPTION = "Identify opportunities with revenue risk factors"
    SUPPORTED_ANALYSIS_TYPES = ["revenue_risk", "financial_risk"]
    
    # Implementation follows the template above...
```

## **Verification**

After adding your agent:

1. **Check Discovery**: Run `python test_modular_rba_architecture.py`
2. **Verify Registration**: Your agent should appear in the registry
3. **Test Execution**: Your analysis types should work automatically
4. **No Code Changes**: The orchestrator should route to your agent without any hardcoding

## **Benefits of This Architecture**

✅ **Zero Hardcoding**: New agents are automatically discovered  
✅ **Modular**: Each agent is independent and focused  
✅ **Extensible**: Add new agents without touching existing code  
✅ **Configuration-Driven**: Business rules in YAML, not Python  
✅ **Type Safety**: Proper validation and error handling  
✅ **Maintainable**: Clear separation of concerns  

## **Common Patterns**

### **Risk Scoring Agents**
```python
def _calculate_risk_score(self, opp, config):
    score = 0
    # Risk factor calculations
    return score
```

### **Threshold-Based Agents**
```python
def _check_threshold(self, value, thresholds):
    if value > thresholds['high']:
        return 'HIGH'
    elif value > thresholds['medium']:
        return 'MEDIUM'
    return 'LOW'
```

### **Time-Based Agents**
```python
def _calculate_days_since(self, date_str):
    try:
        date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
        return (datetime.now() - date).days
    except:
        return 0
```

This architecture ensures that adding new RBA agents is simple, safe, and requires no changes to existing code!
