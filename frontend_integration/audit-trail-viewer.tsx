/**
 * Task 4.3.30: Audit Trail Timeline Viewer Component
 * React component for visualizing audit trail events in timeline format
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  AlertTriangle, 
  Shield, 
  Clock, 
  User, 
  Settings, 
  Download,
  Filter,
  ChevronDown,
  ChevronRight,
  Search
} from 'lucide-react';

// Types
interface AuditEvent {
  event_id: string;
  event_type: 'override' | 'drift_alert' | 'bias_alert' | 'explainability' | 'quarantine' | 'approval';
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  model_id?: string;
  workflow_id?: string;
  step_id?: string;
  user_id?: number;
  approver_id?: number;
  justification?: string;
  technical_details: Record<string, any>;
  tenant_id: string;
  tags: string[];
}

interface AuditTrailResponse {
  events: AuditEvent[];
  total_count: number;
  filtered_count: number;
  query_metadata: Record<string, any>;
}

interface AuditTrailProps {
  tenantId: string;
  apiBaseUrl?: string;
}

// Event type configurations
const EVENT_CONFIG = {
  override: {
    icon: Settings,
    color: 'bg-orange-100 text-orange-800 border-orange-200',
    label: 'Override'
  },
  drift_alert: {
    icon: AlertTriangle,
    color: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    label: 'Drift Alert'
  },
  bias_alert: {
    icon: Shield,
    color: 'bg-red-100 text-red-800 border-red-200',
    label: 'Bias Alert'
  },
  explainability: {
    icon: Clock,
    color: 'bg-blue-100 text-blue-800 border-blue-200',
    label: 'Explanation'
  },
  quarantine: {
    icon: AlertTriangle,
    color: 'bg-red-100 text-red-800 border-red-200',
    label: 'Quarantine'
  },
  approval: {
    icon: User,
    color: 'bg-green-100 text-green-800 border-green-200',
    label: 'Approval'
  }
};

const SEVERITY_CONFIG = {
  low: { color: 'bg-gray-100 text-gray-800', label: 'Low' },
  medium: { color: 'bg-blue-100 text-blue-800', label: 'Medium' },
  high: { color: 'bg-orange-100 text-orange-800', label: 'High' },
  critical: { color: 'bg-red-100 text-red-800', label: 'Critical' }
};

export const AuditTrailViewer: React.FC<AuditTrailProps> = ({ 
  tenantId, 
  apiBaseUrl = 'http://localhost:8000' 
}) => {
  // State
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());
  
  // Filters
  const [timeRange, setTimeRange] = useState('24'); // hours
  const [eventTypeFilter, setEventTypeFilter] = useState<string>('all');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Fetch audit trail data
  const fetchAuditTrail = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        hours: timeRange,
        ...(eventTypeFilter !== 'all' && { event_types: eventTypeFilter }),
        ...(severityFilter !== 'all' && { severity: severityFilter })
      });
      
      const response = await fetch(
        `${apiBaseUrl}/audit-trail/timeline/${tenantId}?${params}`
      );
      
      if (!response.ok) {
        throw new Error(`Failed to fetch audit trail: ${response.statusText}`);
      }
      
      const data: AuditTrailResponse = await response.json();
      setEvents(data.events);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Filter events based on search term
  const filteredEvents = useMemo(() => {
    if (!searchTerm) return events;
    
    const term = searchTerm.toLowerCase();
    return events.filter(event => 
      event.title.toLowerCase().includes(term) ||
      event.description.toLowerCase().includes(term) ||
      event.model_id?.toLowerCase().includes(term) ||
      event.workflow_id?.toLowerCase().includes(term) ||
      event.justification?.toLowerCase().includes(term)
    );
  }, [events, searchTerm]);

  // Load data on mount and filter changes
  useEffect(() => {
    fetchAuditTrail();
  }, [tenantId, timeRange, eventTypeFilter, severityFilter]);

  // Toggle event expansion
  const toggleEventExpansion = (eventId: string) => {
    const newExpanded = new Set(expandedEvents);
    if (newExpanded.has(eventId)) {
      newExpanded.delete(eventId);
    } else {
      newExpanded.add(eventId);
    }
    setExpandedEvents(newExpanded);
  };

  // Export audit trail
  const exportAuditTrail = async (format: 'json' | 'csv') => {
    try {
      const response = await fetch(
        `${apiBaseUrl}/audit-trail/export/${tenantId}?format=${format}&days=${Math.ceil(parseInt(timeRange) / 24)}`
      );
      
      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Download the data
      const blob = new Blob([
        format === 'json' ? JSON.stringify(data.data, null, 2) : data.data
      ], {
        type: format === 'json' ? 'application/json' : 'text/csv'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audit-trail-${tenantId}-${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString()
    };
  };

  // Render event item
  const renderEvent = (event: AuditEvent) => {
    const config = EVENT_CONFIG[event.event_type];
    const severityConfig = SEVERITY_CONFIG[event.severity];
    const IconComponent = config.icon;
    const isExpanded = expandedEvents.has(event.event_id);
    const { date, time } = formatTimestamp(event.timestamp);

    return (
      <div key={event.event_id} className="relative">
        {/* Timeline line */}
        <div className="absolute left-6 top-12 bottom-0 w-0.5 bg-gray-200"></div>
        
        {/* Event card */}
        <Card className="ml-14 mb-4 hover:shadow-md transition-shadow">
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-3">
                {/* Timeline dot */}
                <div className="absolute left-4 w-4 h-4 bg-white border-2 border-gray-300 rounded-full -ml-2"></div>
                
                {/* Event icon */}
                <div className={`p-2 rounded-lg ${config.color} border`}>
                  <IconComponent className="h-4 w-4" />
                </div>
                
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-sm">{event.title}</h3>
                    <Badge variant="outline" className={severityConfig.color}>
                      {severityConfig.label}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{event.description}</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className="text-right text-xs text-gray-500">
                  <div>{date}</div>
                  <div>{time}</div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => toggleEventExpansion(event.event_id)}
                >
                  {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </CardHeader>
          
          {isExpanded && (
            <CardContent className="pt-0">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                {/* Basic info */}
                <div>
                  <h4 className="font-medium mb-2">Event Details</h4>
                  <div className="space-y-1">
                    <div><span className="text-gray-500">Event ID:</span> {event.event_id}</div>
                    <div><span className="text-gray-500">Type:</span> {config.label}</div>
                    {event.model_id && (
                      <div><span className="text-gray-500">Model:</span> {event.model_id}</div>
                    )}
                    {event.workflow_id && (
                      <div><span className="text-gray-500">Workflow:</span> {event.workflow_id}</div>
                    )}
                    {event.user_id && (
                      <div><span className="text-gray-500">User:</span> {event.user_id}</div>
                    )}
                    {event.approver_id && (
                      <div><span className="text-gray-500">Approver:</span> {event.approver_id}</div>
                    )}
                  </div>
                </div>
                
                {/* Technical details */}
                <div>
                  <h4 className="font-medium mb-2">Technical Details</h4>
                  <div className="bg-gray-50 p-3 rounded text-xs font-mono">
                    <pre>{JSON.stringify(event.technical_details, null, 2)}</pre>
                  </div>
                </div>
                
                {/* Justification */}
                {event.justification && (
                  <div className="md:col-span-2">
                    <h4 className="font-medium mb-2">Justification</h4>
                    <p className="text-gray-700 bg-blue-50 p-3 rounded">{event.justification}</p>
                  </div>
                )}
                
                {/* Tags */}
                {event.tags.length > 0 && (
                  <div className="md:col-span-2">
                    <h4 className="font-medium mb-2">Tags</h4>
                    <div className="flex flex-wrap gap-1">
                      {event.tags.map(tag => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          )}
        </Card>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Audit Trail</h2>
          <p className="text-gray-600">Timeline of governance events and decisions</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={() => exportAuditTrail('csv')}>
            <Download className="h-4 w-4 mr-2" />
            Export CSV
          </Button>
          <Button variant="outline" onClick={() => exportAuditTrail('json')}>
            <Download className="h-4 w-4 mr-2" />
            Export JSON
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Filter className="h-5 w-5 mr-2" />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Time range */}
            <div>
              <label className="block text-sm font-medium mb-2">Time Range</label>
              <Select value={timeRange} onValueChange={setTimeRange}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Last Hour</SelectItem>
                  <SelectItem value="24">Last 24 Hours</SelectItem>
                  <SelectItem value="168">Last Week</SelectItem>
                  <SelectItem value="720">Last Month</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Event type */}
            <div>
              <label className="block text-sm font-medium mb-2">Event Type</label>
              <Select value={eventTypeFilter} onValueChange={setEventTypeFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="override">Overrides</SelectItem>
                  <SelectItem value="drift_alert">Drift Alerts</SelectItem>
                  <SelectItem value="bias_alert">Bias Alerts</SelectItem>
                  <SelectItem value="quarantine">Quarantine</SelectItem>
                  <SelectItem value="explainability">Explanations</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Severity */}
            <div>
              <label className="block text-sm font-medium mb-2">Severity</label>
              <Select value={severityFilter} onValueChange={setSeverityFilter}>
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

            {/* Search */}
            <div>
              <label className="block text-sm font-medium mb-2">Search</label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search events..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>
              Timeline ({filteredEvents.length} events)
            </CardTitle>
            <Button variant="outline" size="sm" onClick={fetchAuditTrail}>
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-2">Loading audit trail...</span>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
              <p className="text-red-800">Error: {error}</p>
            </div>
          )}

          {!loading && !error && filteredEvents.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              No audit events found for the selected criteria.
            </div>
          )}

          {!loading && !error && filteredEvents.length > 0 && (
            <div className="relative">
              {filteredEvents.map(renderEvent)}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default AuditTrailViewer;
