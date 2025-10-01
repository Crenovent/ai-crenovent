/**
 * RBA Configuration Panel Component
 * =================================
 * React component for configuring RBA agents with gear button, parameter groups,
 * templates, validation, and real-time preview.
 */

import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Save, 
  RotateCcw, 
  Eye, 
  AlertTriangle, 
  CheckCircle,
  DollarSign,
  Percent,
  ToggleLeft,
  ToggleRight
} from 'lucide-react';

const RBAConfigPanel = ({ agentName, onConfigChange, onClose }) => {
  const [schema, setSchema] = useState(null);
  const [config, setConfig] = useState({});
  const [errors, setErrors] = useState([]);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTemplate, setActiveTemplate] = useState(null);

  useEffect(() => {
    loadAgentSchema();
  }, [agentName]);

  const loadAgentSchema = async () => {
    try {
      const response = await fetch(`/api/rba-config/agents/${agentName}/schema`);
      const data = await response.json();
      
      if (data.success) {
        setSchema(data.schema);
        // Initialize config with defaults
        const defaultConfig = {};
        Object.values(data.schema.parameter_groups).forEach(group => {
          group.forEach(param => {
            defaultConfig[param.name] = param.default;
          });
        });
        setConfig(defaultConfig);
      }
    } catch (error) {
      console.error('Failed to load agent schema:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleParameterChange = (paramName, value) => {
    const newConfig = { ...config, [paramName]: value };
    setConfig(newConfig);
    validateConfig(newConfig);
    generatePreview(newConfig);
  };

  const validateConfig = async (configToValidate) => {
    try {
      const response = await fetch(`/api/rba-config/agents/${agentName}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configToValidate)
      });
      
      const data = await response.json();
      setErrors(data.errors || []);
    } catch (error) {
      console.error('Validation failed:', error);
    }
  };

  const generatePreview = async (configToPreview) => {
    try {
      const response = await fetch(`/api/rba-config/agents/${agentName}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configToPreview)
      });
      
      const data = await response.json();
      setPreview(data);
    } catch (error) {
      console.error('Preview generation failed:', error);
    }
  };

  const applyTemplate = async (templateName) => {
    try {
      const response = await fetch(`/api/rba-config/agents/${agentName}/apply-template`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(templateName)
      });
      
      const data = await response.json();
      if (data.success) {
        setConfig(data.config);
        setActiveTemplate(templateName);
        validateConfig(data.config);
        generatePreview(data.config);
      }
    } catch (error) {
      console.error('Failed to apply template:', error);
    }
  };

  const handleSave = () => {
    if (errors.length === 0) {
      onConfigChange(config);
      onClose();
    }
  };

  const resetToDefaults = () => {
    const defaultConfig = {};
    Object.values(schema.parameter_groups).forEach(group => {
      group.forEach(param => {
        defaultConfig[param.name] = param.default;
      });
    });
    setConfig(defaultConfig);
    setActiveTemplate(null);
    setErrors([]);
    setPreview(null);
  };

  const renderParameterInput = (param) => {
    const value = config[param.name];

    switch (param.ui_component) {
      case 'currency_input':
        return (
          <div className="relative">
            <DollarSign className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <input
              type="number"
              value={value}
              onChange={(e) => handleParameterChange(param.name, parseFloat(e.target.value))}
              className="pl-10 w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              min={param.min}
              max={param.max}
              step="1000"
            />
          </div>
        );

      case 'slider':
        return (
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <input
                type="range"
                value={value}
                onChange={(e) => handleParameterChange(param.name, parseFloat(e.target.value))}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                min={param.min}
                max={param.max}
                step="1"
              />
              <div className="flex items-center space-x-1 min-w-[60px]">
                <span className="text-sm font-medium">{value}</span>
                {param.type === 'percentage' && <Percent className="h-3 w-3 text-gray-400" />}
              </div>
            </div>
          </div>
        );

      case 'checkbox':
        return (
          <button
            onClick={() => handleParameterChange(param.name, !value)}
            className="flex items-center space-x-2"
          >
            {value ? (
              <ToggleRight className="h-6 w-6 text-blue-500" />
            ) : (
              <ToggleLeft className="h-6 w-6 text-gray-400" />
            )}
            <span className={`text-sm ${value ? 'text-blue-600' : 'text-gray-600'}`}>
              {value ? 'Enabled' : 'Disabled'}
            </span>
          </button>
        );

      default:
        return (
          <input
            type="text"
            value={value}
            onChange={(e) => handleParameterChange(param.name, e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          />
        );
    }
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  if (!schema) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6">
          <p>Failed to load configuration schema</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Settings className="h-6 w-6 text-blue-500" />
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  Configure {schema.description}
                </h2>
                <p className="text-sm text-gray-500">
                  {schema.parameter_count} configurable parameters
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
        </div>

        <div className="flex h-[calc(90vh-120px)]">
          {/* Main Configuration Panel */}
          <div className="flex-1 overflow-y-auto p-6">
            {/* Templates */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Quick Templates</h3>
              <div className="grid grid-cols-3 gap-2">
                {schema.templates.map((template) => (
                  <button
                    key={template.name}
                    onClick={() => applyTemplate(template.name)}
                    className={`p-3 text-left border rounded-lg transition-colors ${
                      activeTemplate === template.name
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-medium text-sm">{template.name}</div>
                    <div className="text-xs text-gray-500 mt-1">{template.description}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Parameter Groups */}
            <div className="space-y-6">
              {Object.entries(schema.parameter_groups).map(([groupName, parameters]) => (
                <div key={groupName} className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-900 mb-4">{groupName}</h3>
                  <div className="space-y-4">
                    {parameters.map((param) => (
                      <div key={param.name} className="grid grid-cols-3 gap-4 items-start">
                        <div>
                          <label className="block text-sm font-medium text-gray-700">
                            {param.label}
                          </label>
                          <p className="text-xs text-gray-500 mt-1">{param.description}</p>
                        </div>
                        <div className="col-span-2">
                          {renderParameterInput(param)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Preview Panel */}
          <div className="w-80 border-l border-gray-200 bg-gray-50 p-4 overflow-y-auto">
            <h3 className="text-sm font-medium text-gray-900 mb-3 flex items-center">
              <Eye className="h-4 w-4 mr-2" />
              Configuration Preview
            </h3>

            {/* Validation Status */}
            <div className="mb-4">
              {errors.length === 0 ? (
                <div className="flex items-center text-green-600 text-sm">
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Configuration Valid
                </div>
              ) : (
                <div className="space-y-1">
                  <div className="flex items-center text-red-600 text-sm">
                    <AlertTriangle className="h-4 w-4 mr-2" />
                    {errors.length} Error{errors.length > 1 ? 's' : ''}
                  </div>
                  {errors.map((error, index) => (
                    <div key={index} className="text-xs text-red-600 ml-6">
                      {error}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Impact Preview */}
            {preview && (
              <div className="space-y-3">
                <div className="bg-white rounded p-3">
                  <div className="text-xs font-medium text-gray-700 mb-2">Expected Impact</div>
                  <div className={`text-sm px-2 py-1 rounded ${
                    preview.impact_level === 'high' ? 'bg-red-100 text-red-700' :
                    preview.impact_level === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {preview.impact_level.toUpperCase()} impact expected
                  </div>
                </div>

                {preview.changes && preview.changes.length > 0 && (
                  <div className="bg-white rounded p-3">
                    <div className="text-xs font-medium text-gray-700 mb-2">Key Changes</div>
                    <div className="space-y-1">
                      {preview.changes.slice(0, 3).map((change, index) => (
                        <div key={index} className="text-xs text-gray-600">
                          {change.parameter}: {change.change_percentage}% {change.change_percentage > 0 ? '↑' : '↓'}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {preview.recommendations && (
                  <div className="bg-white rounded p-3">
                    <div className="text-xs font-medium text-gray-700 mb-2">Recommendations</div>
                    <div className="space-y-1">
                      {preview.recommendations.map((rec, index) => (
                        <div key={index} className="text-xs text-gray-600">
                          • {rec}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 p-4 flex items-center justify-between">
          <button
            onClick={resetToDefaults}
            className="flex items-center space-x-2 px-3 py-2 text-gray-600 hover:text-gray-800"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Reset to Defaults</span>
          </button>

          <div className="flex items-center space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={errors.length > 0}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md ${
                errors.length === 0
                  ? 'bg-blue-500 text-white hover:bg-blue-600'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              <Save className="h-4 w-4" />
              <span>Apply Configuration</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RBAConfigPanel;
