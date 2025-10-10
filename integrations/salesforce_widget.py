"""
Salesforce Widget Integration - Task 6.1.56
===========================================
RBIA metrics widget for Salesforce CRM
"""

from typing import Dict, List, Any
import json


class SalesforceRBIAWidget:
    """Salesforce Lightning Web Component integration"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
    
    def generate_lightning_component(self) -> str:
        """Generate Lightning Web Component code"""
        return """
// rbiaMetrics.js - Lightning Web Component
import { LightningElement, api, wire } from 'lwc';
import { getRecord } from 'lightning/uiRecordApi';
import getRBIAMetrics from '@salesforce/apex/RBIAController.getRBIAMetrics';

export default class RbiaMetrics extends LightningElement {
    @api recordId;  // Opportunity or Account ID
    @api objectApiName;
    
    metrics = {};
    loading = true;
    error;
    
    connectedCallback() {
        this.loadMetrics();
    }
    
    async loadMetrics() {
        try {
            this.loading = true;
            
            // Call RBIA API via Apex
            const result = await getRBIAMetrics({
                entityId: this.recordId,
                entityType: this.objectApiName
            });
            
            this.metrics = JSON.parse(result);
            this.error = null;
            
        } catch (error) {
            this.error = error.body.message;
        } finally {
            this.loading = false;
        }
    }
    
    get churnRisk() {
        return this.metrics.churn_prediction?.prediction || 0;
    }
    
    get trustScore() {
        return this.metrics.trust_score?.overall_score || 0;
    }
    
    get riskLevel() {
        const risk = this.churnRisk;
        if (risk > 0.7) return 'High';
        if (risk > 0.4) return 'Medium';
        return 'Low';
    }
}
"""
    
    def generate_apex_controller(self) -> str:
        """Generate Apex controller for API calls"""
        return f"""
// RBIAController.apxc - Apex Controller
public with sharing class RBIAController {{
    
    @AuraEnabled
    public static String getRBIAMetrics(String entityId, String entityType) {{
        
        // Call RBIA API
        HttpRequest req = new HttpRequest();
        req.setEndpoint('{self.api_base_url}/metrics/entity/' + entityId);
        req.setMethod('GET');
        req.setHeader('Content-Type', 'application/json');
        
        Http http = new Http();
        HttpResponse res = http.send(req);
        
        if (res.getStatusCode() == 200) {{
            return res.getBody();
        }} else {{
            throw new AuraHandledException('Failed to fetch RBIA metrics');
        }}
    }}
    
    @AuraEnabled
    public static String getChurnPrediction(String opportunityId) {{
        // Fetch churn prediction for opportunity
        HttpRequest req = new HttpRequest();
        req.setEndpoint('{self.api_base_url}/predict/churn/' + opportunityId);
        req.setMethod('GET');
        
        Http http = new Http();
        HttpResponse res = http.send(req);
        
        return res.getBody();
    }}
}}
"""
    
    def generate_html_template(self) -> str:
        """Generate HTML template for Lightning Component"""
        return """
<!-- rbiaMetrics.html -->
<template>
    <lightning-card title="RBIA Intelligence" icon-name="custom:custom63">
        
        <template if:true={loading}>
            <lightning-spinner alternative-text="Loading..."></lightning-spinner>
        </template>
        
        <template if:true={error}>
            <div class="slds-text-color_error slds-p-around_medium">
                {error}
            </div>
        </template>
        
        <template if:false={loading}>
            <div class="slds-p-around_medium">
                
                <!-- Churn Risk -->
                <div class="metric-card">
                    <div class="slds-text-heading_small">Churn Risk</div>
                    <div class="slds-text-heading_large">{churnRisk}%</div>
                    <lightning-badge label={riskLevel} 
                                   class={riskLevel}></lightning-badge>
                </div>
                
                <!-- Trust Score -->
                <div class="metric-card">
                    <div class="slds-text-heading_small">Trust Score</div>
                    <div class="slds-text-heading_large">{trustScore}</div>
                    <lightning-progress-bar value={trustScore} 
                                          variant="circular"></lightning-progress-bar>
                </div>
                
                <!-- Actions -->
                <div class="slds-p-top_medium">
                    <lightning-button label="View Details" 
                                    onclick={viewDetails}></lightning-button>
                    <lightning-button label="Refresh" 
                                    onclick={loadMetrics}
                                    class="slds-m-left_small"></lightning-button>
                </div>
            </div>
        </template>
        
    </lightning-card>
</template>
"""


class HubSpotRBIAWidget:
    """HubSpot CRM Card integration"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
    
    def generate_crm_card_config(self) -> Dict[str, Any]:
        """Generate HubSpot CRM Card configuration"""
        return {
            'title': 'RBIA Intelligence',
            'fetch': {
                'targetUrl': f'{self.api_base_url}/hubspot/crm-card',
                'objectTypes': [
                    {'name': 'contacts'},
                    {'name': 'companies'},
                    {'name': 'deals'}
                ]
            },
            'display': {
                'properties': [
                    {
                        'name': 'churn_risk',
                        'label': 'Churn Risk',
                        'dataType': 'NUMERIC'
                    },
                    {
                        'name': 'trust_score',
                        'label': 'Trust Score',
                        'dataType': 'NUMERIC'
                    },
                    {
                        'name': 'recommendations',
                        'label': 'Recommendations',
                        'dataType': 'STRING'
                    }
                ]
            }
        }
    
    def generate_webhook_handler(self) -> str:
        """Generate webhook handler for HubSpot"""
        return """
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class HubSpotWebhookPayload(BaseModel):
    objectId: int
    objectType: str
    portalId: int

@app.post("/hubspot/crm-card")
async def hubspot_crm_card_handler(payload: HubSpotWebhookPayload):
    # Fetch RBIA metrics for the entity
    entity_id = payload.objectId
    entity_type = payload.objectType
    
    # Call RBIA services
    metrics = {
        'churn_risk': 0.35,  # Would fetch from RBIA API
        'trust_score': 0.87,
        'recommendations': 'Schedule check-in call'
    }
    
    return {
        'results': [
            {
                'objectId': entity_id,
                'title': 'RBIA Intelligence',
                'properties': [
                    {'label': 'Churn Risk', 'value': f"{metrics['churn_risk']*100:.1f}%"},
                    {'label': 'Trust Score', 'value': f"{metrics['trust_score']*100:.1f}%"},
                    {'label': 'Action', 'value': metrics['recommendations']}
                ]
            }
        ]
    }
"""


def deploy_salesforce_widget():
    """Deploy Salesforce widget"""
    widget = SalesforceRBIAWidget()
    
    # Generate files
    lwc_js = widget.generate_lightning_component()
    apex_ctrl = widget.generate_apex_controller()
    html_template = widget.generate_html_template()
    
    print("Salesforce Widget Generated:")
    print("1. Lightning Web Component (LWC)")
    print("2. Apex Controller")
    print("3. HTML Template")
    print("\nDeploy to Salesforce using SFDX CLI")


def deploy_hubspot_widget():
    """Deploy HubSpot widget"""
    widget = HubSpotRBIAWidget()
    
    config = widget.generate_crm_card_config()
    webhook = widget.generate_webhook_handler()
    
    print("HubSpot Widget Generated:")
    print("1. CRM Card Configuration")
    print("2. Webhook Handler")
    print("\nDeploy via HubSpot Developer Portal")

