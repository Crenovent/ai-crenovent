/**
 * Intelligence Dashboard React Components
 * =====================================
 * 
 * Ready-to-use React/Next.js components for the Intelligence Dashboard system.
 * These components integrate with your Node.js backend to display:
 * - Trust scores with visual indicators
 * - SLA monitoring dashboards
 * - Real-time alerts and notifications
 * - Intelligent recommendations
 * 
 * Usage: Copy these components to your src/components/intelligence/ directory
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, 
  ShieldCheck, 
  ShieldAlert, 
  ShieldX, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  Zap,
  Target,
  Bell,
  Lightbulb,
  Refresh
} from 'lucide-react';

// Types for Intelligence Dashboard data
interface TrustScore {
  capability_id: string;
  overall_score: number;
  trust_level: string;
  trust_level_color: string;
  trust_level_icon: string;
  breakdown: {
    execution: number;
    performance: number;
    compliance: number;
    business_impact: number;
  };
  risk_level: string;
  last_updated: string;
  recommendations: string[];
}

interface SLAMetrics {
  summary_cards: {
    availability: {
      value: number;
      unit: string;
      status: string;
      target: number;
      trend: string;
    };
    latency: {
      value: number;
      unit: string;
      status: string;
      target: number;
      trend: string;
    };
    compliance: {
      value: number;
      unit: string;
      status: string;
      target: number;
      trend: string;
    };
    breaches: {
      value: number;
      unit: string;
      status: string;
      target: number;
      trend: string;
    };
  };
  alerts: {
    critical_count: number;
    warning_count: number;
    total_capabilities: number;
  };
}

interface Alert {
  id: string;
  title: string;
  message: string;
  type: string;
  severity: string;
  capability_id: string;
  triggered_at: string;
  severity_color: string;
  severity_icon: string;
  time_ago: string;
  is_critical: boolean;
  requires_action: boolean;
  recommended_actions: string[];
}

interface Recommendation {
  id: string;
  title: string;
  description: string;
  category: string;
  priority: string;
  type: string;
  capability_id?: string;
  estimated_impact: string;
  effort_required: string;
  actions: string[];
  created_at: string;
}

interface DashboardData {
  summary: {
    total_capabilities: number;
    average_trust_score: number;
    sla_compliance_percentage: number;
    active_alerts_count: number;
    capabilities_at_risk: number;
    high_trust_capabilities: number;
    recommendations_count: number;
  };
  trust_scores: TrustScore[];
  sla_metrics: SLAMetrics;
  active_alerts: Alert[];
  recommendations: Recommendation[];
}

// API service for calling Node.js backend
class IntelligenceAPIService {
  private baseURL: string;

  constructor(baseURL: string = '/api') {
    this.baseURL = baseURL;
  }

  private async fetchWithAuth(endpoint: string, options: RequestInit = {}) {
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      credentials: 'include', // Include cookies for JWT authentication
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  async getDashboard(tenantId: number, timeRangeHours: number = 24): Promise<DashboardData> {
    const result = await this.fetchWithAuth(
      `/intelligence/dashboard?tenant_id=${tenantId}&time_range_hours=${timeRangeHours}&include_trends=true`
    );
    return result.data;
  }

  async getTrustScores(tenantId: number, capabilityIds?: string[], trustLevelFilter?: string): Promise<TrustScore[]> {
    let url = `/intelligence/trust-scores?tenant_id=${tenantId}`;
    if (capabilityIds) url += `&capability_ids=${capabilityIds.join(',')}`;
    if (trustLevelFilter) url += `&trust_level_filter=${trustLevelFilter}`;

    const result = await this.fetchWithAuth(url);
    return result.data.capabilities;
  }

  async getSLADashboard(tenantId: number, slaTier?: string, timeRangeHours: number = 24): Promise<SLAMetrics> {
    let url = `/intelligence/sla-dashboard?tenant_id=${tenantId}&time_range_hours=${timeRangeHours}`;
    if (slaTier) url += `&sla_tier=${slaTier}`;

    const result = await this.fetchWithAuth(url);
    return result.data;
  }

  async calculateTrustScore(tenantId: number, capabilityId: string, lookbackDays: number = 30): Promise<TrustScore> {
    const result = await this.fetchWithAuth('/intelligence/calculate-trust', {
      method: 'POST',
      body: JSON.stringify({
        tenant_id: tenantId,
        capability_id: capabilityId,
        lookback_days: lookbackDays,
      }),
    });
    return result.data;
  }

  async getActiveAlerts(tenantId: number, alertType?: string, severity?: string): Promise<Alert[]> {
    let url = `/intelligence/alerts?tenant_id=${tenantId}`;
    if (alertType) url += `&alert_type=${alertType}`;
    if (severity) url += `&severity=${severity}`;

    const result = await this.fetchWithAuth(url);
    return result.data.alerts;
  }

  async getRecommendations(tenantId: number, category?: string, priority?: string): Promise<Recommendation[]> {
    let url = `/intelligence/recommendations?tenant_id=${tenantId}`;
    if (category) url += `&category=${category}`;
    if (priority) url += `&priority=${priority}`;

    const result = await this.fetchWithAuth(url);
    return result.data.recommendations;
  }

  async getHealthCheck(): Promise<any> {
    return this.fetchWithAuth('/intelligence/health');
  }
}

// Global API service instance
const intelligenceAPI = new IntelligenceAPIService();

// Trust Score Component
export const TrustScoreCard: React.FC<{ trustScore: TrustScore }> = ({ trustScore }) => {
  const getTrustIcon = (level: string) => {
    switch (level) {
      case 'critical': return <ShieldCheck className="h-5 w-5" style={{ color: trustScore.trust_level_color }} />;
      case 'high': return <Shield className="h-5 w-5" style={{ color: trustScore.trust_level_color }} />;
      case 'medium': return <ShieldAlert className="h-5 w-5" style={{ color: trustScore.trust_level_color }} />;
      case 'low': return <ShieldAlert className="h-5 w-5" style={{ color: trustScore.trust_level_color }} />;
      default: return <ShieldX className="h-5 w-5" style={{ color: trustScore.trust_level_color }} />;
    }
  };

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">
          {trustScore.capability_id}
        </CardTitle>
        {getTrustIcon(trustScore.trust_level)}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold" style={{ color: trustScore.trust_level_color }}>
          {(trustScore.overall_score * 100).toFixed(1)}%
        </div>
        <div className="flex items-center space-x-2 mt-2">
          <Badge variant="outline" style={{ borderColor: trustScore.trust_level_color, color: trustScore.trust_level_color }}>
            {trustScore.trust_level.toUpperCase()}
          </Badge>
          <Badge variant="secondary">
            {trustScore.risk_level} risk
          </Badge>
        </div>
        
        <div className="mt-4 space-y-2">
          <div className="flex justify-between text-xs">
            <span>Execution</span>
            <span>{(trustScore.breakdown.execution * 100).toFixed(0)}%</span>
          </div>
          <Progress value={trustScore.breakdown.execution * 100} className="h-1" />
          
          <div className="flex justify-between text-xs">
            <span>Performance</span>
            <span>{(trustScore.breakdown.performance * 100).toFixed(0)}%</span>
          </div>
          <Progress value={trustScore.breakdown.performance * 100} className="h-1" />
          
          <div className="flex justify-between text-xs">
            <span>Compliance</span>
            <span>{(trustScore.breakdown.compliance * 100).toFixed(0)}%</span>
          </div>
          <Progress value={trustScore.breakdown.compliance * 100} className="h-1" />
        </div>

        {trustScore.recommendations.length > 0 && (
          <div className="mt-3">
            <div className="text-xs text-muted-foreground mb-1">Top Recommendation:</div>
            <div className="text-xs bg-muted p-2 rounded">
              {trustScore.recommendations[0]}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// SLA Metrics Component
export const SLAMetricsCard: React.FC<{ slaMetrics: SLAMetrics }> = ({ slaMetrics }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-600';
      case 'good': return 'text-blue-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'declining': return <TrendingDown className="h-4 w-4 text-red-600" />;
      default: return <div className="h-4 w-4" />; // Placeholder for stable
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Availability</CardTitle>
          <Target className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className={`text-2xl font-bold ${getStatusColor(slaMetrics.summary_cards.availability.status)}`}>
            {slaMetrics.summary_cards.availability.value.toFixed(2)}%
          </div>
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              Target: {slaMetrics.summary_cards.availability.target}%
            </p>
            {getTrendIcon(slaMetrics.summary_cards.availability.trend)}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Latency</CardTitle>
          <Zap className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className={`text-2xl font-bold ${getStatusColor(slaMetrics.summary_cards.latency.status)}`}>
            {slaMetrics.summary_cards.latency.value.toFixed(1)}ms
          </div>
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              Target: {slaMetrics.summary_cards.latency.target}ms
            </p>
            {getTrendIcon(slaMetrics.summary_cards.latency.trend)}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Compliance</CardTitle>
          <CheckCircle className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className={`text-2xl font-bold ${getStatusColor(slaMetrics.summary_cards.compliance.status)}`}>
            {slaMetrics.summary_cards.compliance.value.toFixed(1)}%
          </div>
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              Target: {slaMetrics.summary_cards.compliance.target}%
            </p>
            {getTrendIcon(slaMetrics.summary_cards.compliance.trend)}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Alerts</CardTitle>
          <Bell className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-red-600">
            {slaMetrics.alerts.critical_count}
          </div>
          <p className="text-xs text-muted-foreground">
            {slaMetrics.alerts.warning_count} warnings, {slaMetrics.alerts.total_capabilities} total capabilities
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

// Alerts Component
export const AlertsPanel: React.FC<{ alerts: Alert[] }> = ({ alerts }) => {
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-600" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      default: return <Bell className="h-4 w-4 text-blue-600" />;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bell className="h-5 w-5" />
          Active Alerts ({alerts.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {alerts.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-2 text-green-600" />
              <p>No active alerts</p>
              <p className="text-xs">All systems are operating normally</p>
            </div>
          ) : (
            alerts.map((alert) => (
              <Alert key={alert.id} className={`border-l-4 ${
                alert.severity === 'critical' ? 'border-l-red-500' :
                alert.severity === 'warning' ? 'border-l-yellow-500' :
                'border-l-blue-500'
              }`}>
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-2">
                    {getSeverityIcon(alert.severity)}
                    <div>
                      <h4 className="text-sm font-medium">{alert.title}</h4>
                      <AlertDescription className="text-xs">
                        {alert.message}
                      </AlertDescription>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline" className="text-xs">
                          {alert.capability_id}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {alert.time_ago}
                        </span>
                      </div>
                    </div>
                  </div>
                  {alert.requires_action && (
                    <Button size="sm" variant="outline">
                      Action Required
                    </Button>
                  )}
                </div>
              </Alert>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Recommendations Component
export const RecommendationsPanel: React.FC<{ recommendations: Recommendation[] }> = ({ recommendations }) => {
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-600 border-red-200';
      case 'medium': return 'text-yellow-600 border-yellow-200';
      default: return 'text-green-600 border-green-200';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Lightbulb className="h-5 w-5" />
          Intelligent Recommendations ({recommendations.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {recommendations.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-2 text-green-600" />
              <p>No recommendations</p>
              <p className="text-xs">System is performing optimally</p>
            </div>
          ) : (
            recommendations.map((rec) => (
              <div key={rec.id} className={`border rounded-lg p-3 ${getPriorityColor(rec.priority)}`}>
                <div className="flex items-start justify-between mb-2">
                  <h4 className="text-sm font-medium">{rec.title}</h4>
                  <Badge variant="outline" className={getPriorityColor(rec.priority)}>
                    {rec.priority}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mb-2">
                  {rec.description}
                </p>
                <div className="flex items-center gap-2 text-xs">
                  <span className="bg-muted px-2 py-1 rounded">
                    Impact: {rec.estimated_impact}
                  </span>
                  <span className="bg-muted px-2 py-1 rounded">
                    Effort: {rec.effort_required}
                  </span>
                  {rec.capability_id && (
                    <span className="bg-muted px-2 py-1 rounded">
                      {rec.capability_id}
                    </span>
                  )}
                </div>
                {rec.actions.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs font-medium mb-1">Recommended Actions:</p>
                    <ul className="text-xs space-y-1">
                      {rec.actions.slice(0, 2).map((action, index) => (
                        <li key={index} className="flex items-start gap-1">
                          <span className="text-muted-foreground">•</span>
                          <span>{action}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Main Intelligence Dashboard Component
export const IntelligenceDashboard: React.FC<{ tenantId: number }> = ({ tenantId }) => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  const loadDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await intelligenceAPI.getDashboard(tenantId);
      setDashboardData(data);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  }, [tenantId]);

  useEffect(() => {
    loadDashboardData();
    
    // Auto-refresh every 5 minutes
    const interval = setInterval(loadDashboardData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [loadDashboardData]);

  if (loading && !dashboardData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
          <p className="mt-2 text-sm text-muted-foreground">Loading intelligence dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="border-red-200">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Failed to load intelligence dashboard: {error}
          <Button size="sm" variant="outline" onClick={loadDashboardData} className="ml-2">
            <Refresh className="h-4 w-4 mr-1" />
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (!dashboardData) {
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Header with Summary */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Intelligence Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time capability monitoring and insights
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm text-muted-foreground">Last updated</p>
            <p className="text-sm font-medium">{lastUpdated.toLocaleTimeString()}</p>
          </div>
          <Button onClick={loadDashboardData} disabled={loading}>
            <Refresh className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Capabilities</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardData.summary.total_capabilities}</div>
            <p className="text-xs text-muted-foreground">
              {dashboardData.summary.high_trust_capabilities} high trust
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Trust Score</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(dashboardData.summary.average_trust_score * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Across all capabilities
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">SLA Compliance</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dashboardData.summary.sla_compliance_percentage.toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {dashboardData.summary.capabilities_at_risk} at risk
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <Bell className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {dashboardData.summary.active_alerts_count}
            </div>
            <p className="text-xs text-muted-foreground">
              {dashboardData.summary.recommendations_count} recommendations
            </p>
          </CardContent>
        </Card>
      </div>

      {/* SLA Metrics */}
      <div>
        <h2 className="text-xl font-semibold mb-4">SLA Performance</h2>
        <SLAMetricsCard slaMetrics={dashboardData.sla_metrics} />
      </div>

      {/* Trust Scores Grid */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Trust Scores</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {dashboardData.trust_scores.slice(0, 8).map((trustScore) => (
            <TrustScoreCard key={trustScore.capability_id} trustScore={trustScore} />
          ))}
        </div>
      </div>

      {/* Alerts and Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AlertsPanel alerts={dashboardData.active_alerts} />
        <RecommendationsPanel recommendations={dashboardData.recommendations} />
      </div>
    </div>
  );
};

export default IntelligenceDashboard;
