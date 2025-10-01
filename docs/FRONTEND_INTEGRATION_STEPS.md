# 🚀 Frontend Integration Steps - Add Gear Buttons to Workflow Cards

## 🎯 Current Status
- ✅ **API Working**: `/api/rba-config/agents` returns 2 agents successfully
- ✅ **Registry Fixed**: `ownerless_deals_detection` now supported
- ✅ **React Components Ready**: `WorkflowConfigButton.jsx` and `RBAConfigPanel.jsx` created
- ❌ **Frontend Integration**: Need to add components to your existing UI

## 📋 **Step-by-Step Integration**

### **Step 1: Copy React Components to Your Frontend**

Copy these files to your frontend project:
```
frontend_integration/WorkflowConfigButton.jsx  → your-frontend/src/components/
frontend_integration/RBAConfigPanel.jsx        → your-frontend/src/components/
```

### **Step 2: Install Required Dependencies**

```bash
npm install lucide-react  # For icons (Settings, Save, etc.)
```

### **Step 3: Add Gear Button to Workflow Cards**

Find your existing workflow card component and add the gear button:

```jsx
import React, { useState } from 'react';
import WorkflowConfigButton from './WorkflowConfigButton';

// Your existing workflow card component
const WorkflowCard = ({ workflow, onExecute }) => {
  const [workflowConfig, setWorkflowConfig] = useState({});

  const handleConfigUpdate = (newConfig) => {
    setWorkflowConfig(newConfig);
    console.log('Config updated:', newConfig);
  };

  const handleExecute = () => {
    // Pass config to execution
    onExecute(workflow.name, workflowConfig);
  };

  return (
    <div className="workflow-card border rounded-lg p-4">
      {/* Header with gear button */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold">{workflow.name}</h3>
        
        {/* 🔧 ADD THIS GEAR BUTTON */}
        <WorkflowConfigButton
          workflowName={workflow.name}
          agentName={workflow.agent_name} // e.g., "sandbagging_detection"
          currentConfig={workflowConfig}
          onConfigUpdate={handleConfigUpdate}
          size="md"
        />
      </div>

      {/* Existing workflow content */}
      <p className="text-gray-600 mb-4">{workflow.description}</p>
      
      <button 
        onClick={handleExecute}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Execute Workflow
      </button>
    </div>
  );
};
```

### **Step 4: Map Workflow Names to Agent Names**

Create a mapping object to connect your workflow names to RBA agent names:

```jsx
// Add this mapping object
const WORKFLOW_TO_AGENT_MAP = {
  'Sandbagging Detection': 'sandbagging_detection',
  'Pipeline Summary': 'pipeline_summary',
  'Ownerless Deals Detection': 'ownerless_deals',
  'Pipeline Hygiene': 'stale_deals',
  'Missing Fields Audit': 'missing_fields',
  'Duplicate Detection': 'duplicate_detection',
  'Deal Risk Scoring': 'deal_risk_scoring',
  // Add more mappings as needed
};

// Use in your component
const agentName = WORKFLOW_TO_AGENT_MAP[workflow.name] || workflow.name.toLowerCase().replace(/ /g, '_');
```

### **Step 5: Update Pipeline Execution to Use Config**

Modify your pipeline execution to pass the configuration:

```jsx
const executeWorkflow = async (workflowName, config = {}) => {
  try {
    const response = await fetch('/api/pipeline/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_input: `Execute ${workflowName}`,
        context: {
          timestamp: new Date().toISOString(),
          source: 'pipeline_agents_ui',
          workflow_config: config  // 🔧 Pass config here
        }
      })
    });

    const result = await response.json();
    
    if (result.success) {
      console.log('✅ Workflow executed with config:', config);
      console.log('📊 Results:', result);
      // Handle results...
    } else {
      console.error('❌ Workflow execution failed:', result);
    }
  } catch (error) {
    console.error('❌ Network error:', error);
  }
};
```

## 🎯 **Quick Test Steps**

### **Test 1: Check API Access**
Open browser console and run:
```javascript
fetch('/api/rba-config/agents')
  .then(r => r.json())
  .then(data => console.log('Available agents:', data));
```

### **Test 2: Test Configuration Schema**
```javascript
fetch('/api/rba-config/agents/sandbagging_detection/schema')
  .then(r => r.json())
  .then(data => console.log('Sandbagging config schema:', data));
```

### **Test 3: Verify Gear Button**
1. Add `WorkflowConfigButton` to any workflow card
2. Click the gear button
3. Configuration panel should open with parameter groups

## 🔧 **Troubleshooting**

### **If Gear Button Doesn't Appear:**
1. Check browser console for import errors
2. Verify `lucide-react` is installed
3. Check if `agentName` prop is correctly set

### **If Configuration Panel Doesn't Open:**
1. Check network tab for API calls to `/api/rba-config/`
2. Verify the server is running on localhost:8000
3. Check CORS settings if accessing from different port

### **If Configuration Doesn't Apply:**
1. Verify `onConfigUpdate` is called with new config
2. Check that config is passed to `workflow_config` in API request
3. Look for config merge logs in server terminal

## 📊 **Expected Results**

After integration, you should see:
- ⚙️ **Gear button** on each workflow card
- 🔵 **Blue indicator** when custom config is applied
- 🎛️ **Configuration panel** with parameter groups
- ✅ **Real-time validation** and error messages
- 📊 **Configuration templates** (Conservative/Aggressive/Balanced)
- 🔄 **Config applied** to workflow execution

## 🚀 **Next Steps**

1. **Copy components** to your frontend
2. **Add gear button** to one workflow card first
3. **Test configuration** with sandbagging detection
4. **Expand to all workflow cards** once working
5. **Add more agents** to CONFIG_SCHEMAS in `rba_config_api.py`

The system is **ready and working** - you just need to integrate the React components into your existing UI! 🎉
