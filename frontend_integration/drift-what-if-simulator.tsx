/**
 * Task 4.3.40: Drift What-If Simulator Component
 * React component for running drift stress scenarios and simulations
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Play, 
  AlertTriangle, 
  TrendingUp, 
  Shield, 
  Zap,
  FileText,
  RefreshCw,
  Download
} from 'lucide-react';

// Types
interface DriftSimulationRequest {
  simulation_id?: string;
  tenant_id: string;
  model_id: string;
  scenario: string;
  severity: string;
  duration_hours: number;
  baseline_data: Record<string, number[]>;
  sample_size: number;
  drift_intensity: number;
  noise_level: number;
  conditions: Record<string, any>;
  description: string;
  created_by: string;
  tags: string[];
}

interface DriftSimulationResult {
  simulation_id: string;
  tenant_id: string;
  model_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at?: string;
  execution_time_ms?: number;
  drift_detected: boolean;
  drift_alerts: any[];
  drift_scores: Record<string, number>;
  original_data: Record<string, any>;
  simulated_data: Record<string, any>;
  impact_assessment: Record<string, any>;
  recommendations: string[];
  quarantine_triggered: boolean;
  quarantine_reason?: string;
}

interface DriftSimulatorProps {
  tenantId: string;
  apiBaseUrl?: string;
}

const SCENARIOS = {
  data_shift: { label: 'Data Shift', icon: TrendingUp, description: 'Mean shift in feature distributions' },
  concept_drift: { label: 'Concept Drift', icon: TrendingUp, description: 'Gradual change in relationships' },
  covariate_shift: { label: 'Covariate Shift', icon: TrendingUp, description: 'Change in feature variance' },
  feature_corruption: { label: 'Feature Corruption', icon: AlertTriangle, description: 'Noise and data quality issues' },
  seasonal_change: { label: 'Seasonal Change', icon: RefreshCw, description: 'Periodic variations' },
  population_shift: { label: 'Population Shift', icon: Shield, description: 'Population characteristics change' },
  adversarial_attack: { label: 'Adversarial Attack', icon: Zap, description: 'Targeted perturbations' }
};

const SEVERITIES = {
  mild: { label: 'Mild', color: 'bg-green-100 text-green-800' },
  moderate: { label: 'Moderate', color: 'bg-yellow-100 text-yellow-800' },
  severe: { label: 'Severe', color: 'bg-orange-100 text-orange-800' },
  extreme: { label: 'Extreme', color: 'bg-red-100 text-red-800' }
};

export const DriftWhatIfSimulator: React.FC<DriftSimulatorProps> = ({ 
  tenantId, 
  apiBaseUrl = 'http://localhost:8002' 
}) => {
  // State
  const [simulations, setSimulations] = useState<DriftSimulationResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedResult, setSelectedResult] = useState<DriftSimulationResult | null>(null);
  
  // Form state
  const [modelId, setModelId] = useState('');
  const [scenario, setScenario] = useState('data_shift');
  const [severity, setSeverity] = useState('moderate');
  const [description, setDescription] = useState('');
  const [baselineData, setBaselineData] = useState('{"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]}');
  const [sampleSize, setSampleSize] = useState('1000');
  const [driftIntensity, setDriftIntensity] = useState('0.1');

  // Fetch simulations
  const fetchSimulations = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/drift-simulation/tenant/${tenantId}`);
      if (!response.ok) throw new Error(`Failed to fetch: ${response.statusText}`);
      
      const data = await response.json();
      setSimulations(data.simulations || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch simulations');
    }
  };

  // Run simulation
  const runSimulation = async () => {
    if (!modelId) {
      setError('Model ID is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Parse baseline data
      let parsedBaselineData;
      try {
        parsedBaselineData = JSON.parse(baselineData);
      } catch (e) {
        throw new Error('Invalid baseline data JSON format');
      }

      const request: DriftSimulationRequest = {
        tenant_id: tenantId,
        model_id: modelId,
        scenario,
        severity,
        duration_hours: 24,
        baseline_data: parsedBaselineData,
        sample_size: parseInt(sampleSize),
        drift_intensity: parseFloat(driftIntensity),
        noise_level: 0.05,
        conditions: {},
        description: description || `${SCENARIOS[scenario as keyof typeof SCENARIOS].label} simulation`,
        created_by: 'user',
        tags: [scenario, severity]
      };

      const response = await fetch(`${apiBaseUrl}/drift-simulation/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`Simulation failed: ${response.statusText}`);
      }

      const result: DriftSimulationResult = await response.json();
      setSimulations(prev => [result, ...prev]);
      setSelectedResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  // Run batch simulations
  const runBatchSimulations = async () => {
    if (!modelId) {
      setError('Model ID is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let parsedBaselineData;
      try {
        parsedBaselineData = JSON.parse(baselineData);
      } catch (e) {
        throw new Error('Invalid baseline data JSON format');
      }

      const response = await fetch(`${apiBaseUrl}/drift-simulation/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: tenantId,
          model_id: modelId,
          baseline_data: parsedBaselineData,
          created_by: 'user'
        })
      });

      if (!response.ok) {
        throw new Error(`Batch simulation failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Refresh simulations list
      await fetchSimulations();
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch simulation failed');
    } finally {
      setLoading(false);
    }
  };

  // Load simulations on mount
  useEffect(() => {
    fetchSimulations();
  }, [tenantId]);

  // Render simulation result
  const renderSimulationResult = (result: DriftSimulationResult) => {
    const scenarioConfig = SCENARIOS[result.model_id.split('_')[0] as keyof typeof SCENARIOS] || SCENARIOS.data_shift;
    const IconComponent = scenarioConfig.icon;
    
    return (
      <Card 
        key={result.simulation_id}
        className={`cursor-pointer transition-all hover:shadow-md ${
          selectedResult?.simulation_id === result.simulation_id ? 'ring-2 ring-blue-500' : ''
        }`}
        onClick={() => setSelectedResult(result)}
      >
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <IconComponent className="h-4 w-4" />
              <div>
                <div className="font-medium text-sm">{result.model_id}</div>
                <div className="text-xs text-gray-500">
                  {new Date(result.started_at).toLocaleString()}
                </div>
              </div>
            </div>
            <div className="flex space-x-1">
              <Badge variant={result.status === 'completed' ? 'default' : 'secondary'}>
                {result.status}
              </Badge>
              {result.drift_detected && (
                <Badge variant="destructive">Drift</Badge>
              )}
              {result.quarantine_triggered && (
                <Badge variant="destructive">Quarantine</Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="text-sm space-y-1">
            <div>Alerts: {result.drift_alerts.length}</div>
            <div>Execution: {result.execution_time_ms}ms</div>
            {result.impact_assessment.risk_level && (
              <div>Risk: 
                <Badge variant="outline" className="ml-1">
                  {result.impact_assessment.risk_level}
                </Badge>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  // Render result details
  const renderResultDetails = () => {
    if (!selectedResult) return null;

    return (
      <Card>
        <CardHeader>
          <CardTitle>Simulation Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Status */}
            <div>
              <h4 className="font-medium mb-2">Status</h4>
              <div className="flex space-x-2">
                <Badge variant={selectedResult.status === 'completed' ? 'default' : 'secondary'}>
                  {selectedResult.status}
                </Badge>
                {selectedResult.drift_detected && <Badge variant="destructive">Drift Detected</Badge>}
                {selectedResult.quarantine_triggered && <Badge variant="destructive">Quarantine Triggered</Badge>}
              </div>
            </div>

            {/* Drift Scores */}
            {Object.keys(selectedResult.drift_scores).length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Drift Scores</h4>
                <div className="space-y-1">
                  {Object.entries(selectedResult.drift_scores).map(([feature, score]) => (
                    <div key={feature} className="flex justify-between text-sm">
                      <span>{feature}:</span>
                      <span className={score > 0.3 ? 'text-red-600' : score > 0.15 ? 'text-yellow-600' : 'text-green-600'}>
                        {score.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Impact Assessment */}
            {selectedResult.impact_assessment && (
              <div>
                <h4 className="font-medium mb-2">Impact Assessment</h4>
                <div className="bg-gray-50 p-3 rounded text-sm">
                  <div>Risk Level: <Badge variant="outline">{selectedResult.impact_assessment.risk_level}</Badge></div>
                  <div>Business Impact: {selectedResult.impact_assessment.business_impact}</div>
                  <div>Urgency: {selectedResult.impact_assessment.urgency}</div>
                  {selectedResult.impact_assessment.confidence_degradation && (
                    <div>Confidence Degradation: {(selectedResult.impact_assessment.confidence_degradation * 100).toFixed(1)}%</div>
                  )}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {selectedResult.recommendations.length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Recommendations</h4>
                <div className="space-y-1">
                  {selectedResult.recommendations.map((rec, idx) => (
                    <div key={idx} className="text-sm bg-blue-50 p-2 rounded">
                      {rec}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Quarantine Info */}
            {selectedResult.quarantine_triggered && (
              <div>
                <h4 className="font-medium mb-2">Quarantine</h4>
                <div className="bg-red-50 border border-red-200 p-3 rounded text-sm">
                  <div className="font-medium text-red-800">Model would be quarantined</div>
                  {selectedResult.quarantine_reason && (
                    <div className="text-red-700 mt-1">{selectedResult.quarantine_reason}</div>
                  )}
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
          <h2 className="text-2xl font-bold">Drift What-If Simulator</h2>
          <p className="text-gray-600">Simulate stress conditions and drift scenarios</p>
        </div>
        <Button variant="outline" onClick={fetchSimulations}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Simulation Form */}
      <Card>
        <CardHeader>
          <CardTitle>Run Simulation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Model ID */}
            <div>
              <label className="block text-sm font-medium mb-2">Model ID</label>
              <Input
                placeholder="Enter model ID..."
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
              />
            </div>

            {/* Scenario */}
            <div>
              <label className="block text-sm font-medium mb-2">Stress Scenario</label>
              <Select value={scenario} onValueChange={setScenario}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(SCENARIOS).map(([key, config]) => (
                    <SelectItem key={key} value={key}>
                      {config.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Severity */}
            <div>
              <label className="block text-sm font-medium mb-2">Severity</label>
              <Select value={severity} onValueChange={setSeverity}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(SEVERITIES).map(([key, config]) => (
                    <SelectItem key={key} value={key}>
                      {config.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Sample Size */}
            <div>
              <label className="block text-sm font-medium mb-2">Sample Size</label>
              <Input
                type="number"
                placeholder="1000"
                value={sampleSize}
                onChange={(e) => setSampleSize(e.target.value)}
              />
            </div>

            {/* Drift Intensity */}
            <div>
              <label className="block text-sm font-medium mb-2">Drift Intensity</label>
              <Input
                type="number"
                step="0.01"
                placeholder="0.1"
                value={driftIntensity}
                onChange={(e) => setDriftIntensity(e.target.value)}
              />
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium mb-2">Description</label>
              <Input
                placeholder="Optional description..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>

            {/* Baseline Data */}
            <div className="md:col-span-2">
              <label className="block text-sm font-medium mb-2">Baseline Data (JSON)</label>
              <Textarea
                placeholder='{"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]}'
                value={baselineData}
                onChange={(e) => setBaselineData(e.target.value)}
                rows={3}
              />
            </div>
          </div>

          <div className="flex space-x-2 mt-4">
            <Button onClick={runSimulation} disabled={loading}>
              <Play className="h-4 w-4 mr-2" />
              {loading ? 'Running...' : 'Run Simulation'}
            </Button>
            <Button variant="outline" onClick={runBatchSimulations} disabled={loading}>
              <Zap className="h-4 w-4 mr-2" />
              Run Batch
            </Button>
          </div>

          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-md p-4">
              <p className="text-red-800">Error: {error}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Simulations List */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle>Recent Simulations ({simulations.length})</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {simulations.length === 0 ? (
                  <p className="text-gray-500 text-center py-4">No simulations yet</p>
                ) : (
                  simulations.map(renderSimulationResult)
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Result Details */}
        <div>
          {selectedResult ? (
            renderResultDetails()
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center py-8">
                <p className="text-gray-500">Select a simulation to view details</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default DriftWhatIfSimulator;
