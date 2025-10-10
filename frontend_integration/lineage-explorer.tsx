/**
 * Task 4.3.38: Lineage Explorer Component
 * React component for visualizing end-to-end data lineage in governance context
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Database, 
  Brain, 
  GitBranch, 
  AlertTriangle, 
  FileText, 
  Settings,
  Search,
  Filter,
  Download,
  Maximize2,
  RefreshCw
} from 'lucide-react';

// Types
interface LineageNode {
  node_id: string;
  node_type: 'dataset' | 'model' | 'workflow' | 'step' | 'decision' | 'override' | 'evidence' | 'explanation';
  name: string;
  description: string;
  created_at: string;
  created_by?: string;
  tenant_id: string;
  metadata: Record<string, any>;
  inputs: string[];
  outputs: string[];
  governance_status: 'compliant' | 'flagged' | 'overridden' | 'quarantined';
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
}

interface LineageEdge {
  edge_id: string;
  source_node_id: string;
  target_node_id: string;
  relationship_type: string;
  created_at: string;
  tenant_id: string;
  metadata: Record<string, any>;
}

interface LineageGraph {
  graph_id: string;
  tenant_id: string;
  nodes: LineageNode[];
  edges: LineageEdge[];
  root_node_id?: string;
  created_at: string;
  query_metadata: Record<string, any>;
}

interface LineageExplorerProps {
  tenantId: string;
  apiBaseUrl?: string;
}

// Node type configurations
const NODE_CONFIG = {
  dataset: {
    icon: Database,
    color: 'bg-blue-100 text-blue-800 border-blue-200',
    label: 'Dataset'
  },
  model: {
    icon: Brain,
    color: 'bg-purple-100 text-purple-800 border-purple-200',
    label: 'Model'
  },
  workflow: {
    icon: GitBranch,
    color: 'bg-green-100 text-green-800 border-green-200',
    label: 'Workflow'
  },
  step: {
    icon: GitBranch,
    color: 'bg-teal-100 text-teal-800 border-teal-200',
    label: 'Step'
  },
  decision: {
    icon: AlertTriangle,
    color: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    label: 'Decision'
  },
  override: {
    icon: Settings,
    color: 'bg-orange-100 text-orange-800 border-orange-200',
    label: 'Override'
  },
  evidence: {
    icon: FileText,
    color: 'bg-gray-100 text-gray-800 border-gray-200',
    label: 'Evidence'
  },
  explanation: {
    icon: FileText,
    color: 'bg-indigo-100 text-indigo-800 border-indigo-200',
    label: 'Explanation'
  }
};

const RISK_CONFIG = {
  low: { color: 'bg-green-100 text-green-800', label: 'Low' },
  medium: { color: 'bg-yellow-100 text-yellow-800', label: 'Medium' },
  high: { color: 'bg-orange-100 text-orange-800', label: 'High' },
  critical: { color: 'bg-red-100 text-red-800', label: 'Critical' }
};

const STATUS_CONFIG = {
  compliant: { color: 'bg-green-100 text-green-800', label: 'Compliant' },
  flagged: { color: 'bg-yellow-100 text-yellow-800', label: 'Flagged' },
  overridden: { color: 'bg-orange-100 text-orange-800', label: 'Overridden' },
  quarantined: { color: 'bg-red-100 text-red-800', label: 'Quarantined' }
};

export const LineageExplorer: React.FC<LineageExplorerProps> = ({ 
  tenantId, 
  apiBaseUrl = 'http://localhost:8001' 
}) => {
  // State
  const [graph, setGraph] = useState<LineageGraph | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<LineageNode | null>(null);
  
  // Query state
  const [queryType, setQueryType] = useState<'recent' | 'model' | 'workflow'>('recent');
  const [modelId, setModelId] = useState('');
  const [workflowId, setWorkflowId] = useState('');
  const [maxDepth, setMaxDepth] = useState('3');
  
  // Filters
  const [nodeTypeFilter, setNodeTypeFilter] = useState<string>('all');
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // Fetch lineage data
  const fetchLineage = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      let url = '';
      
      switch (queryType) {
        case 'model':
          if (!modelId) {
            setError('Model ID is required');
            return;
          }
          url = `${apiBaseUrl}/lineage/model/${modelId}?tenant_id=${tenantId}&depth=${maxDepth}`;
          break;
        case 'workflow':
          if (!workflowId) {
            setError('Workflow ID is required');
            return;
          }
          url = `${apiBaseUrl}/lineage/workflow/${workflowId}?tenant_id=${tenantId}&depth=${maxDepth}`;
          break;
        case 'recent':
        default:
          url = `${apiBaseUrl}/lineage/decisions/${tenantId}?hours=24&include_overrides=true`;
          break;
      }
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch lineage: ${response.statusText}`);
      }
      
      const data: LineageGraph = await response.json();
      setGraph(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  }, [tenantId, queryType, modelId, workflowId, maxDepth, apiBaseUrl]);

  // Filter nodes based on current filters
  const filteredNodes = useMemo(() => {
    if (!graph) return [];
    
    return graph.nodes.filter(node => {
      if (nodeTypeFilter !== 'all' && node.node_type !== nodeTypeFilter) return false;
      if (riskFilter !== 'all' && node.risk_level !== riskFilter) return false;
      if (statusFilter !== 'all' && node.governance_status !== statusFilter) return false;
      return true;
    });
  }, [graph, nodeTypeFilter, riskFilter, statusFilter]);

  // Load data on mount and query changes
  useEffect(() => {
    fetchLineage();
  }, [fetchLineage]);

  // Export lineage data
  const exportLineage = async (format: 'json' | 'svg') => {
    if (!graph) return;
    
    try {
      if (format === 'json') {
        const blob = new Blob([JSON.stringify(graph, null, 2)], {
          type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `lineage-${tenantId}-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
      // SVG export would require additional visualization library
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    }
  };

  // Render node card
  const renderNode = (node: LineageNode) => {
    const config = NODE_CONFIG[node.node_type];
    const riskConfig = RISK_CONFIG[node.risk_level];
    const statusConfig = STATUS_CONFIG[node.governance_status];
    const IconComponent = config.icon;
    
    return (
      <Card 
        key={node.node_id} 
        className={`cursor-pointer transition-all hover:shadow-md ${
          selectedNode?.node_id === node.node_id ? 'ring-2 ring-blue-500' : ''
        }`}
        onClick={() => setSelectedNode(node)}
      >
        <CardHeader className="pb-2">
          <div className="flex items-center space-x-2">
            <div className={`p-2 rounded-lg ${config.color} border`}>
              <IconComponent className="h-4 w-4" />
            </div>
            <div className="flex-1">
              <CardTitle className="text-sm">{node.name}</CardTitle>
              <p className="text-xs text-gray-600 mt-1">{node.description}</p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="flex flex-wrap gap-1 mb-2">
            <Badge variant="outline" className={config.color}>
              {config.label}
            </Badge>
            <Badge variant="outline" className={riskConfig.color}>
              {riskConfig.label}
            </Badge>
            <Badge variant="outline" className={statusConfig.color}>
              {statusConfig.label}
            </Badge>
          </div>
          
          {/* Connection indicators */}
          <div className="flex justify-between text-xs text-gray-500">
            <span>↑ {node.inputs.length} inputs</span>
            <span>↓ {node.outputs.length} outputs</span>
          </div>
          
          {/* Tags */}
          {node.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {node.tags.slice(0, 3).map(tag => (
                <Badge key={tag} variant="secondary" className="text-xs">
                  {tag}
                </Badge>
              ))}
              {node.tags.length > 3 && (
                <Badge variant="secondary" className="text-xs">
                  +{node.tags.length - 3}
                </Badge>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  // Render node details panel
  const renderNodeDetails = () => {
    if (!selectedNode) return null;
    
    const config = NODE_CONFIG[selectedNode.node_type];
    const IconComponent = config.icon;
    
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <IconComponent className="h-5 w-5" />
            <span>{selectedNode.name}</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Basic info */}
            <div>
              <h4 className="font-medium mb-2">Basic Information</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><span className="text-gray-500">ID:</span> {selectedNode.node_id}</div>
                <div><span className="text-gray-500">Type:</span> {config.label}</div>
                <div><span className="text-gray-500">Status:</span> 
                  <Badge variant="outline" className={`ml-1 ${STATUS_CONFIG[selectedNode.governance_status].color}`}>
                    {STATUS_CONFIG[selectedNode.governance_status].label}
                  </Badge>
                </div>
                <div><span className="text-gray-500">Risk:</span>
                  <Badge variant="outline" className={`ml-1 ${RISK_CONFIG[selectedNode.risk_level].color}`}>
                    {RISK_CONFIG[selectedNode.risk_level].label}
                  </Badge>
                </div>
                <div><span className="text-gray-500">Created:</span> {new Date(selectedNode.created_at).toLocaleDateString()}</div>
                {selectedNode.created_by && (
                  <div><span className="text-gray-500">Created by:</span> {selectedNode.created_by}</div>
                )}
              </div>
            </div>
            
            {/* Description */}
            <div>
              <h4 className="font-medium mb-2">Description</h4>
              <p className="text-sm text-gray-700">{selectedNode.description}</p>
            </div>
            
            {/* Metadata */}
            {Object.keys(selectedNode.metadata).length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Metadata</h4>
                <div className="bg-gray-50 p-3 rounded text-xs font-mono">
                  <pre>{JSON.stringify(selectedNode.metadata, null, 2)}</pre>
                </div>
              </div>
            )}
            
            {/* Connections */}
            <div>
              <h4 className="font-medium mb-2">Connections</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Inputs ({selectedNode.inputs.length}):</span>
                  <div className="mt-1 space-y-1">
                    {selectedNode.inputs.slice(0, 3).map(input => (
                      <div key={input} className="text-xs bg-blue-50 px-2 py-1 rounded">
                        {input}
                      </div>
                    ))}
                    {selectedNode.inputs.length > 3 && (
                      <div className="text-xs text-gray-500">+{selectedNode.inputs.length - 3} more</div>
                    )}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Outputs ({selectedNode.outputs.length}):</span>
                  <div className="mt-1 space-y-1">
                    {selectedNode.outputs.slice(0, 3).map(output => (
                      <div key={output} className="text-xs bg-green-50 px-2 py-1 rounded">
                        {output}
                      </div>
                    ))}
                    {selectedNode.outputs.length > 3 && (
                      <div className="text-xs text-gray-500">+{selectedNode.outputs.length - 3} more</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Tags */}
            {selectedNode.tags.length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Tags</h4>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.tags.map(tag => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Lineage Explorer</h2>
          <p className="text-gray-600">End-to-end data lineage for governance</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={() => exportLineage('json')}>
            <Download className="h-4 w-4 mr-2" />
            Export JSON
          </Button>
          <Button variant="outline" onClick={fetchLineage}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Query controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Search className="h-5 w-5 mr-2" />
            Query
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Query type */}
            <div>
              <label className="block text-sm font-medium mb-2">Query Type</label>
              <Select value={queryType} onValueChange={(value: any) => setQueryType(value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="recent">Recent Decisions</SelectItem>
                  <SelectItem value="model">By Model ID</SelectItem>
                  <SelectItem value="workflow">By Workflow ID</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Model ID input */}
            {queryType === 'model' && (
              <div>
                <label className="block text-sm font-medium mb-2">Model ID</label>
                <Input
                  placeholder="Enter model ID..."
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                />
              </div>
            )}

            {/* Workflow ID input */}
            {queryType === 'workflow' && (
              <div>
                <label className="block text-sm font-medium mb-2">Workflow ID</label>
                <Input
                  placeholder="Enter workflow ID..."
                  value={workflowId}
                  onChange={(e) => setWorkflowId(e.target.value)}
                />
              </div>
            )}

            {/* Max depth */}
            <div>
              <label className="block text-sm font-medium mb-2">Max Depth</label>
              <Select value={maxDepth} onValueChange={setMaxDepth}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2">2 levels</SelectItem>
                  <SelectItem value="3">3 levels</SelectItem>
                  <SelectItem value="4">4 levels</SelectItem>
                  <SelectItem value="5">5 levels</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Filter className="h-5 w-5 mr-2" />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Node type filter */}
            <div>
              <label className="block text-sm font-medium mb-2">Node Type</label>
              <Select value={nodeTypeFilter} onValueChange={setNodeTypeFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="dataset">Datasets</SelectItem>
                  <SelectItem value="model">Models</SelectItem>
                  <SelectItem value="decision">Decisions</SelectItem>
                  <SelectItem value="override">Overrides</SelectItem>
                  <SelectItem value="evidence">Evidence</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Risk filter */}
            <div>
              <label className="block text-sm font-medium mb-2">Risk Level</label>
              <Select value={riskFilter} onValueChange={setRiskFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Status filter */}
            <div>
              <label className="block text-sm font-medium mb-2">Governance Status</label>
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Statuses</SelectItem>
                  <SelectItem value="compliant">Compliant</SelectItem>
                  <SelectItem value="flagged">Flagged</SelectItem>
                  <SelectItem value="overridden">Overridden</SelectItem>
                  <SelectItem value="quarantined">Quarantined</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Graph view */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>
                Lineage Graph ({filteredNodes.length} nodes)
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loading && (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  <span className="ml-2">Loading lineage...</span>
                </div>
              )}

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-md p-4">
                  <p className="text-red-800">Error: {error}</p>
                </div>
              )}

              {!loading && !error && filteredNodes.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No lineage nodes found for the selected criteria.
                </div>
              )}

              {!loading && !error && filteredNodes.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
                  {filteredNodes.map(renderNode)}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Node details */}
        <div>
          {selectedNode ? (
            renderNodeDetails()
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center py-8">
                <p className="text-gray-500">Select a node to view details</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Graph metadata */}
      {graph && (
        <Card>
          <CardHeader>
            <CardTitle>Graph Metadata</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div><span className="text-gray-500">Graph ID:</span> {graph.graph_id}</div>
              <div><span className="text-gray-500">Total Nodes:</span> {graph.nodes.length}</div>
              <div><span className="text-gray-500">Total Edges:</span> {graph.edges.length}</div>
              <div><span className="text-gray-500">Created:</span> {new Date(graph.created_at).toLocaleString()}</div>
            </div>
            
            {graph.query_metadata && Object.keys(graph.query_metadata).length > 0 && (
              <div className="mt-4">
                <h4 className="font-medium mb-2">Query Details</h4>
                <div className="bg-gray-50 p-3 rounded text-xs font-mono">
                  <pre>{JSON.stringify(graph.query_metadata, null, 2)}</pre>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default LineageExplorer;
