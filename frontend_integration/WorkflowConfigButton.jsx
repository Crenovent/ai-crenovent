/**
 * Workflow Configuration Button Component
 * ======================================
 * Gear button component that opens the RBA configuration panel for any workflow.
 * Integrates with existing workflow cards and provides visual feedback.
 */

import React, { useState, useEffect } from 'react';
import { Settings, Check, AlertCircle, Loader } from 'lucide-react';
import RBAConfigPanel from './RBAConfigPanel';

const WorkflowConfigButton = ({ 
  workflowName, 
  agentName, 
  currentConfig = {}, 
  onConfigUpdate,
  className = "",
  size = "md" 
}) => {
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [configStatus, setConfigStatus] = useState('default'); // default, custom, loading, error
  const [hasCustomConfig, setHasCustomConfig] = useState(false);

  useEffect(() => {
    // Check if current config differs from defaults
    checkConfigStatus();
  }, [currentConfig, agentName]);

  const checkConfigStatus = async () => {
    if (!agentName || Object.keys(currentConfig).length === 0) {
      setConfigStatus('default');
      setHasCustomConfig(false);
      return;
    }

    try {
      setConfigStatus('loading');
      
      // Get default config from schema
      const response = await fetch(`/api/rba-config/agents/${agentName}/schema`);
      const data = await response.json();
      
      if (data.success) {
        const defaultConfig = {};
        Object.values(data.schema.parameter_groups).forEach(group => {
          group.forEach(param => {
            defaultConfig[param.name] = param.default;
          });
        });

        // Compare with current config
        const isCustom = Object.keys(currentConfig).some(key => 
          currentConfig[key] !== defaultConfig[key]
        );

        setHasCustomConfig(isCustom);
        setConfigStatus(isCustom ? 'custom' : 'default');
      } else {
        setConfigStatus('error');
      }
    } catch (error) {
      console.error('Failed to check config status:', error);
      setConfigStatus('error');
    }
  };

  const handleConfigChange = (newConfig) => {
    if (onConfigUpdate) {
      onConfigUpdate(newConfig);
    }
    setIsConfigOpen(false);
    
    // Update status after config change
    setTimeout(checkConfigStatus, 100);
  };

  const getButtonStyle = () => {
    const baseClasses = "inline-flex items-center justify-center rounded-full transition-all duration-200 hover:scale-110";
    
    const sizeClasses = {
      sm: "w-6 h-6",
      md: "w-8 h-8", 
      lg: "w-10 h-10"
    };

    const statusClasses = {
      default: "bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700",
      custom: "bg-blue-100 text-blue-600 hover:bg-blue-200 ring-2 ring-blue-300",
      loading: "bg-gray-100 text-gray-400 cursor-not-allowed",
      error: "bg-red-100 text-red-500 hover:bg-red-200"
    };

    return `${baseClasses} ${sizeClasses[size]} ${statusClasses[configStatus]} ${className}`;
  };

  const getIconSize = () => {
    const iconSizes = {
      sm: "h-3 w-3",
      md: "h-4 w-4",
      lg: "h-5 w-5"
    };
    return iconSizes[size];
  };

  const renderIcon = () => {
    switch (configStatus) {
      case 'loading':
        return <Loader className={`${getIconSize()} animate-spin`} />;
      case 'error':
        return <AlertCircle className={getIconSize()} />;
      case 'custom':
        return (
          <div className="relative">
            <Settings className={getIconSize()} />
            <Check className="absolute -top-1 -right-1 h-2 w-2 text-blue-600" />
          </div>
        );
      default:
        return <Settings className={getIconSize()} />;
    }
  };

  const getTooltipText = () => {
    switch (configStatus) {
      case 'loading':
        return 'Checking configuration...';
      case 'error':
        return 'Configuration error';
      case 'custom':
        return 'Custom configuration applied';
      default:
        return 'Configure workflow parameters';
    }
  };

  return (
    <>
      <div className="relative group">
        <button
          onClick={() => setIsConfigOpen(true)}
          disabled={configStatus === 'loading'}
          className={getButtonStyle()}
          title={getTooltipText()}
        >
          {renderIcon()}
        </button>

        {/* Tooltip */}
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 text-xs text-white bg-gray-900 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-10">
          {getTooltipText()}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
        </div>

        {/* Custom config indicator */}
        {hasCustomConfig && (
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full border-2 border-white">
            <div className="w-full h-full bg-blue-500 rounded-full animate-pulse"></div>
          </div>
        )}
      </div>

      {/* Configuration Panel */}
      {isConfigOpen && (
        <RBAConfigPanel
          agentName={agentName}
          onConfigChange={handleConfigChange}
          onClose={() => setIsConfigOpen(false)}
        />
      )}
    </>
  );
};

export default WorkflowConfigButton;
