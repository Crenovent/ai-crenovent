# RBA Configuration UI Integration Guide

## 🎯 Overview

This guide shows how to integrate the RBA configuration system with your existing frontend. The system provides:

- **Gear button** for each workflow card
- **100+ configurable parameters** per agent
- **Real-time validation** and preview
- **Configuration templates** (Conservative, Aggressive, Balanced)
- **Visual feedback** for custom configurations

## 🚀 Quick Start

### 1. Add Configuration Button to Workflow Cards

```jsx
import WorkflowConfigButton from './WorkflowConfigButton';

// In your existing workflow card component
<div className="workflow-card">
  <div className="card-header">
    <h3>{workflowName}</h3>
    {/* Add the gear button */}
    <WorkflowConfigButton
      workflowName={workflowName}
      agentName={agentName} // e.g., "sandbagging_detection"
      currentConfig={workflowConfig}
      onConfigUpdate={handleConfigUpdate}
      size="md"
    />
  </div>
  <div className="card-content">
    {/* Your existing workflow content */}
  </div>
</div>
```

### 2. Handle Configuration Updates

```jsx
const handleConfigUpdate = (newConfig) => {
  // Save config to your state/context
  setWorkflowConfig(newConfig);
  
  // Optionally persist to backend
  saveWorkflowConfig(workflowName, newConfig);
  
  // Re-run analysis with new config
  executeWorkflow(workflowName, newConfig);
};
```

### 3. Pass Configuration to Pipeline Execution

```jsx
// When executing pipeline agents
const executeWorkflow = async (workflowName, config) => {
  const response = await fetch('/api/pipeline/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_input: `Execute ${workflowName}`,
      context: {
        workflow_config: config // Pass config here
      }
    })
  });
  
  const result = await response.json();
  // Handle result...
};
```

## 📋 Available Agents and Their Parameters

### Sandbagging Detection (`sandbagging_detection`)
- **Core Thresholds**: `high_value_threshold`, `sandbagging_threshold`, `low_probability_threshold`
- **Advanced Settings**: `mega_deal_threshold`, `confidence_threshold`, `advanced_stage_probability_threshold`
- **Industry Adjustments**: `enable_industry_adjustments`

### Pipeline Summary (`pipeline_summary`)
- **Analysis Scope**: `include_closed_deals`, `include_stage_breakdown`, `include_owner_breakdown`
- **Calculation Settings**: `probability_weighted`, `minimum_deal_amount`

### More Agents (Add as needed)
```javascript
// Example of adding new agent configuration
const AGENT_CONFIGS = {
  'pipeline_hygiene': {
    parameters: [
      'stale_deal_threshold_days',
      'require_close_date',
      'require_amount',
      'minimum_deal_amount'
    ]
  },
  'deal_risk_scoring': {
    parameters: [
      'high_risk_threshold',
      'probability_weight',
      'amount_weight',
      'activity_weight'
    ]
  }
};
```

## 🎨 Styling and Customization

### Button Styles
```jsx
// Small gear button for compact cards
<WorkflowConfigButton size="sm" className="ml-auto" />

// Large gear button for featured workflows  
<WorkflowConfigButton size="lg" className="shadow-lg" />

// Custom styling
<WorkflowConfigButton 
  className="bg-purple-100 text-purple-600 hover:bg-purple-200"
/>
```

### Configuration Panel Theming
The configuration panel uses Tailwind CSS classes. You can customize:

```css
/* Custom theme variables */
:root {
  --config-primary: #3b82f6;
  --config-secondary: #6b7280;
  --config-success: #10b981;
  --config-warning: #f59e0b;
  --config-error: #ef4444;
}
```

## 🔧 API Endpoints

The configuration system provides these REST endpoints:

### Get Available Agents
```javascript
GET /api/rba-config/agents
// Returns: { success: true, agents: [...], total_count: N }
```

### Get Agent Configuration Schema
```javascript
GET /api/rba-config/agents/{agent_name}/schema
// Returns: { success: true, schema: { parameter_groups: {...}, templates: [...] } }
```

### Validate Configuration
```javascript
POST /api/rba-config/agents/{agent_name}/validate
// Body: { "high_value_threshold": 500000, "sandbagging_threshold": 75 }
// Returns: { success: true, validated_config: {...}, errors: [] }
```

### Preview Configuration Impact
```javascript
POST /api/rba-config/agents/{agent_name}/preview
// Body: { "high_value_threshold": 100000 }
// Returns: { success: true, impact_level: "high", changes: [...], recommendations: [...] }
```

### Apply Configuration Template
```javascript
POST /api/rba-config/agents/{agent_name}/apply-template
// Body: "Conservative"
// Returns: { success: true, config: {...}, description: "..." }
```

## 🎯 Integration with Existing Workflow Cards

### Option 1: Modify Existing Cards
```jsx
// Add to your existing workflow card component
const WorkflowCard = ({ workflow, onExecute }) => {
  const [config, setConfig] = useState({});
  
  return (
    <div className="workflow-card">
      <div className="flex items-center justify-between">
        <h3>{workflow.name}</h3>
        <WorkflowConfigButton
          agentName={workflow.agent_name}
          currentConfig={config}
          onConfigUpdate={setConfig}
        />
      </div>
      <button onClick={() => onExecute(workflow.name, config)}>
        Execute
      </button>
    </div>
  );
};
```

### Option 2: Wrapper Component
```jsx
// Create a wrapper that adds config to any workflow
const ConfigurableWorkflow = ({ children, agentName }) => {
  const [config, setConfig] = useState({});
  
  return (
    <div className="relative">
      {children}
      <div className="absolute top-2 right-2">
        <WorkflowConfigButton
          agentName={agentName}
          currentConfig={config}
          onConfigUpdate={setConfig}
        />
      </div>
    </div>
  );
};
```

## 📊 Result Visualization Integration

The configuration system also defines result schemas for different visualization types:

```javascript
// Example result schema from API
{
  "result_schema": {
    "primary_visualizations": ["kpi_cards", "risk_distribution_pie", "top_deals_table"],
    "drill_down_fields": ["opportunity_id", "account_id", "owner_id"],
    "export_formats": ["csv", "excel", "pdf"]
  }
}
```

You can use this to dynamically render appropriate charts:

```jsx
const renderVisualization = (data, schema) => {
  return schema.primary_visualizations.map(vizType => {
    switch (vizType) {
      case 'kpi_cards':
        return <KPICards data={data.summary_metrics} />;
      case 'risk_distribution_pie':
        return <PieChart data={data.risk_distribution} />;
      case 'top_deals_table':
        return <DataTable data={data.flagged_deals} drillDown={schema.drill_down_fields} />;
      default:
        return null;
    }
  });
};
```

## 🔒 State Management

### Using React Context
```jsx
// Create configuration context
const ConfigContext = createContext();

const ConfigProvider = ({ children }) => {
  const [workflowConfigs, setWorkflowConfigs] = useState({});
  
  const updateConfig = (workflowName, config) => {
    setWorkflowConfigs(prev => ({
      ...prev,
      [workflowName]: config
    }));
  };
  
  return (
    <ConfigContext.Provider value={{ workflowConfigs, updateConfig }}>
      {children}
    </ConfigContext.Provider>
  );
};
```

### Using Redux/Zustand
```javascript
// Redux slice for configurations
const configSlice = createSlice({
  name: 'config',
  initialState: { workflowConfigs: {} },
  reducers: {
    updateWorkflowConfig: (state, action) => {
      const { workflowName, config } = action.payload;
      state.workflowConfigs[workflowName] = config;
    }
  }
});
```

## 🧪 Testing

### Unit Tests
```jsx
import { render, screen, fireEvent } from '@testing-library/react';
import WorkflowConfigButton from './WorkflowConfigButton';

test('opens configuration panel on click', () => {
  render(
    <WorkflowConfigButton 
      agentName="sandbagging_detection"
      onConfigUpdate={jest.fn()}
    />
  );
  
  fireEvent.click(screen.getByRole('button'));
  expect(screen.getByText('Configure')).toBeInTheDocument();
});
```

### Integration Tests
```javascript
// Test configuration API integration
test('validates configuration correctly', async () => {
  const config = { high_value_threshold: 500000 };
  const response = await fetch('/api/rba-config/agents/sandbagging_detection/validate', {
    method: 'POST',
    body: JSON.stringify(config)
  });
  
  const result = await response.json();
  expect(result.success).toBe(true);
});
```

## 🚀 Deployment

1. **Copy Components**: Copy `RBAConfigPanel.jsx` and `WorkflowConfigButton.jsx` to your components directory
2. **Install Dependencies**: Ensure you have `lucide-react` for icons
3. **Update API**: The backend API endpoints are automatically available at `/api/rba-config/`
4. **Test Integration**: Start with one workflow card to test the integration

## 🎉 Next Steps

1. **Add more agents** to the `AGENT_REGISTRY` in `rba_config_api.py`
2. **Create custom parameter types** for complex configurations
3. **Add result visualization** components based on result schemas
4. **Implement configuration persistence** in your database
5. **Add user-specific and tenant-specific** default configurations

The system is designed to be **modular and extensible** - you can easily add new agents, parameter types, and visualization components as needed!
