#!/usr/bin/env python3
"""
Chapter 14.1 Capability Registry Population
==========================================

Implements Build Plan Epic 1 - Chapter 14.1: Capability Registry & Marketplace
Populates the empty registry with baseline templates for RBA, RBIA, and AALA

Critical Tasks:
- 14.1.6: ARR, churn, QBR workflows (SaaS overlay)
- 14.1.7: Credit scoring, AML, fraud (Banking overlay)
- 14.1.8: Claims lifecycle, underwriting (Insurance overlay)
- 14.1.10: CRUD schema objects (Governed access)
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.services.connection_pool_manager import ConnectionPoolManager

logger = logging.getLogger(__name__)

# =============================================================================
# SAAS CAPABILITY TEMPLATES (Task 14.1.6)
# =============================================================================

SAAS_RBA_TEMPLATES = [
    {
        'name': 'ARR Pipeline Hygiene',
        'description': 'Automated pipeline data quality checks and standardization',
        'category': 'pipeline_management',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'fetch_opportunities',
                    'type': 'query',
                    'params': {
                        'source': 'salesforce',
                        'query': 'SELECT Id, Name, Amount, StageName, CloseDate FROM Opportunity WHERE Amount > 0',
                        'filters': ['tenant_id']
                    }
                },
                {
                    'id': 'validate_pipeline_data',
                    'type': 'decision',
                    'params': {
                        'rules': [
                            {'field': 'Amount', 'operator': '>', 'value': 0},
                            {'field': 'CloseDate', 'operator': 'not_null'},
                            {'field': 'StageName', 'operator': 'in', 'value': ['Prospecting', 'Qualification', 'Proposal', 'Closed Won', 'Closed Lost']}
                        ]
                    }
                },
                {
                    'id': 'notify_data_issues',
                    'type': 'notify',
                    'params': {
                        'channel': 'slack',
                        'message': 'Pipeline data quality issues detected: {validation_errors}',
                        'recipients': ['revops@company.com']
                    }
                }
            ]
        }
    },
    {
        'name': 'ARR Forecasting',
        'description': 'Automated ARR forecast calculation and reporting',
        'category': 'forecasting',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'calculate_arr_metrics',
                    'type': 'query',
                    'params': {
                        'source': 'postgres',
                        'query': '''
                            SELECT 
                                SUM(CASE WHEN StageName = 'Closed Won' THEN Amount * 12 ELSE 0 END) as current_arr,
                                SUM(CASE WHEN StageName IN ('Proposal', 'Negotiation') THEN Amount * 12 * Probability/100 ELSE 0 END) as pipeline_arr
                            FROM opportunities 
                            WHERE tenant_id = $1
                        '''
                    }
                },
                {
                    'id': 'generate_forecast_report',
                    'type': 'decision',
                    'params': {
                        'rules': [
                            {'condition': 'current_arr > previous_arr', 'action': 'positive_growth'},
                            {'condition': 'pipeline_arr < target_arr * 0.8', 'action': 'pipeline_alert'}
                        ]
                    }
                }
            ]
        }
    },
    {
        'name': 'Churn Risk Detection',
        'description': 'Identify accounts at risk of churning based on usage patterns',
        'category': 'customer_success',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'analyze_usage_patterns',
                    'type': 'query',
                    'params': {
                        'source': 'usage_analytics',
                        'query': 'SELECT account_id, last_login, feature_usage_score, support_tickets FROM usage_metrics WHERE tenant_id = $1'
                    }
                },
                {
                    'id': 'calculate_churn_score',
                    'type': 'decision',
                    'params': {
                        'rules': [
                            {'field': 'last_login', 'operator': '>', 'value': '30 days', 'score': 20},
                            {'field': 'feature_usage_score', 'operator': '<', 'value': 0.3, 'score': 30},
                            {'field': 'support_tickets', 'operator': '>', 'value': 5, 'score': 25}
                        ]
                    }
                },
                {
                    'id': 'trigger_retention_workflow',
                    'type': 'notify',
                    'params': {
                        'condition': 'churn_score > 50',
                        'channel': 'customer_success_team',
                        'message': 'High churn risk detected for account: {account_name}'
                    }
                }
            ]
        }
    },
    {
        'name': 'QBR Preparation Automation',
        'description': 'Automated quarterly business review data compilation',
        'category': 'account_management',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'compile_account_metrics',
                    'type': 'query',
                    'params': {
                        'source': 'multiple',
                        'queries': {
                            'revenue': 'SELECT SUM(amount) FROM opportunities WHERE account_id = $1 AND close_date >= $2',
                            'usage': 'SELECT avg_monthly_usage FROM usage_stats WHERE account_id = $1',
                            'support': 'SELECT COUNT(*) FROM support_tickets WHERE account_id = $1 AND created_date >= $2'
                        }
                    }
                },
                {
                    'id': 'generate_qbr_report',
                    'type': 'decision',
                    'params': {
                        'template': 'qbr_template',
                        'metrics': ['revenue_growth', 'usage_trends', 'support_health', 'renewal_probability']
                    }
                }
            ]
        }
    },
    {
        'name': 'MRR Tracking and Alerts',
        'description': 'Monthly Recurring Revenue monitoring with automated alerts',
        'category': 'revenue_operations',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'calculate_mrr_metrics',
                    'type': 'query',
                    'params': {
                        'source': 'billing_system',
                        'query': '''
                            SELECT 
                                SUM(monthly_amount) as current_mrr,
                                SUM(CASE WHEN status = 'new' THEN monthly_amount ELSE 0 END) as new_mrr,
                                SUM(CASE WHEN status = 'churned' THEN -monthly_amount ELSE 0 END) as churn_mrr
                            FROM subscriptions 
                            WHERE tenant_id = $1 AND month = $2
                        '''
                    }
                },
                {
                    'id': 'mrr_trend_analysis',
                    'type': 'decision',
                    'params': {
                        'rules': [
                            {'condition': 'current_mrr < previous_mrr * 0.95', 'action': 'decline_alert'},
                            {'condition': 'churn_mrr > new_mrr', 'action': 'churn_alert'},
                            {'condition': 'current_mrr > target_mrr', 'action': 'target_achieved'}
                        ]
                    }
                }
            ]
        }
    }
]

SAAS_RBIA_TEMPLATES = [
    {
        'name': 'Intelligent Lead Scoring',
        'description': 'ML-powered lead scoring with behavioral analysis',
        'category': 'lead_management',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'extract_lead_features',
                    'type': 'query',
                    'params': {
                        'source': 'crm',
                        'features': ['company_size', 'industry', 'website_activity', 'email_engagement', 'demo_requests']
                    }
                },
                {
                    'id': 'ml_lead_scoring',
                    'type': 'ml_decision',
                    'params': {
                        'model': 'lead_conversion_predictor',
                        'features': ['company_size', 'industry', 'engagement_score'],
                        'threshold': 0.7
                    }
                },
                {
                    'id': 'route_qualified_leads',
                    'type': 'decision',
                    'params': {
                        'rules': [
                            {'condition': 'lead_score > 0.8', 'action': 'assign_to_senior_rep'},
                            {'condition': 'lead_score > 0.6', 'action': 'assign_to_standard_rep'},
                            {'condition': 'lead_score < 0.4', 'action': 'nurture_sequence'}
                        ]
                    }
                }
            ]
        }
    },
    {
        'name': 'Dynamic Pricing Optimization',
        'description': 'AI-driven pricing recommendations based on market data',
        'category': 'pricing_strategy',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'market_analysis',
                    'type': 'query',
                    'params': {
                        'source': 'market_intelligence',
                        'query': 'SELECT competitor_pricing, market_segment, customer_willingness_to_pay FROM market_data'
                    }
                },
                {
                    'id': 'pricing_optimization',
                    'type': 'ml_decision',
                    'params': {
                        'model': 'pricing_optimizer',
                        'inputs': ['current_pricing', 'competitor_pricing', 'customer_segment', 'usage_patterns'],
                        'optimization_goal': 'maximize_revenue'
                    }
                }
            ]
        }
    },
    {
        'name': 'Predictive Churn Analysis',
        'description': 'Advanced ML model for predicting customer churn',
        'category': 'customer_retention',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'feature_engineering',
                    'type': 'query',
                    'params': {
                        'features': ['usage_decline_rate', 'support_ticket_sentiment', 'payment_delays', 'feature_adoption_rate']
                    }
                },
                {
                    'id': 'churn_prediction',
                    'type': 'ml_decision',
                    'params': {
                        'model': 'churn_predictor_v2',
                        'prediction_horizon': '90_days',
                        'threshold': 0.75
                    }
                },
                {
                    'id': 'intervention_strategy',
                    'type': 'decision',
                    'params': {
                        'strategies': {
                            'high_risk': 'executive_outreach',
                            'medium_risk': 'success_manager_call',
                            'low_risk': 'automated_nurture'
                        }
                    }
                }
            ]
        }
    }
]

SAAS_AALA_TEMPLATES = [
    {
        'name': 'Conversational Revenue Assistant',
        'description': 'AI agent for natural language revenue queries and analysis',
        'category': 'revenue_intelligence',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'intent_recognition',
                    'type': 'agent_call',
                    'params': {
                        'agent': 'revenue_query_agent',
                        'capabilities': ['data_analysis', 'trend_identification', 'forecasting']
                    }
                },
                {
                    'id': 'data_retrieval_planning',
                    'type': 'agent_call',
                    'params': {
                        'agent': 'data_orchestrator',
                        'task': 'plan_data_queries',
                        'sources': ['salesforce', 'billing', 'usage_analytics']
                    }
                },
                {
                    'id': 'insight_generation',
                    'type': 'agent_call',
                    'params': {
                        'agent': 'insight_synthesizer',
                        'task': 'generate_business_insights',
                        'context': 'revenue_performance'
                    }
                }
            ]
        }
    },
    {
        'name': 'Autonomous Deal Review Agent',
        'description': 'AI agent that autonomously reviews and flags deal risks',
        'category': 'deal_management',
        'industry_tags': ['SaaS'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'deal_analysis',
                    'type': 'agent_call',
                    'params': {
                        'agent': 'deal_analyzer',
                        'analysis_areas': ['pricing_anomalies', 'timeline_risks', 'stakeholder_engagement']
                    }
                },
                {
                    'id': 'risk_assessment',
                    'type': 'agent_call',
                    'params': {
                        'agent': 'risk_evaluator',
                        'risk_factors': ['competitive_pressure', 'budget_constraints', 'decision_maker_involvement']
                    }
                },
                {
                    'id': 'recommendation_engine',
                    'type': 'agent_call',
                    'params': {
                        'agent': 'strategy_advisor',
                        'task': 'generate_action_plan',
                        'context': 'deal_acceleration'
                    }
                }
            ]
        }
    }
]

# =============================================================================
# BANKING CAPABILITY TEMPLATES (Task 14.1.7)
# =============================================================================

BANKING_RBA_TEMPLATES = [
    {
        'name': 'Credit Score Validation',
        'description': 'Automated credit score checks and validation workflow',
        'category': 'credit_assessment',
        'industry_tags': ['Banking'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'fetch_credit_data',
                    'type': 'query',
                    'params': {
                        'source': 'credit_bureau',
                        'query': 'SELECT credit_score, payment_history, debt_ratio FROM credit_report WHERE customer_id = $1'
                    }
                },
                {
                    'id': 'validate_credit_criteria',
                    'type': 'decision',
                    'params': {
                        'rules': [
                            {'field': 'credit_score', 'operator': '>=', 'value': 650, 'weight': 40},
                            {'field': 'debt_ratio', 'operator': '<', 'value': 0.4, 'weight': 30},
                            {'field': 'payment_history', 'operator': '>', 'value': 0.9, 'weight': 30}
                        ]
                    }
                },
                {
                    'id': 'compliance_check',
                    'type': 'governance',
                    'params': {
                        'policy_pack': 'RBI_LENDING_GUIDELINES',
                        'evidence_required': True
                    }
                }
            ]
        }
    },
    {
        'name': 'AML Transaction Monitoring',
        'description': 'Anti-Money Laundering transaction pattern analysis',
        'category': 'compliance',
        'industry_tags': ['Banking'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'transaction_analysis',
                    'type': 'query',
                    'params': {
                        'source': 'transaction_db',
                        'query': 'SELECT amount, frequency, counterparty FROM transactions WHERE customer_id = $1 AND date >= $2'
                    }
                },
                {
                    'id': 'aml_pattern_detection',
                    'type': 'decision',
                    'params': {
                        'suspicious_patterns': [
                            'large_cash_deposits',
                            'frequent_small_transactions',
                            'unusual_geographic_patterns',
                            'high_risk_counterparties'
                        ]
                    }
                },
                {
                    'id': 'regulatory_reporting',
                    'type': 'notify',
                    'params': {
                        'condition': 'suspicious_activity_detected',
                        'recipients': ['compliance@bank.com', 'regulator@rbi.gov.in'],
                        'report_type': 'SAR'
                    }
                }
            ]
        }
    },
    {
        'name': 'Fraud Detection Workflow',
        'description': 'Real-time fraud detection and prevention',
        'category': 'fraud_prevention',
        'industry_tags': ['Banking'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'transaction_risk_scoring',
                    'type': 'decision',
                    'params': {
                        'risk_factors': [
                            'unusual_location',
                            'high_amount',
                            'off_hours_transaction',
                            'merchant_risk_category'
                        ]
                    }
                },
                {
                    'id': 'fraud_action',
                    'type': 'decision',
                    'params': {
                        'rules': [
                            {'condition': 'risk_score > 80', 'action': 'block_transaction'},
                            {'condition': 'risk_score > 60', 'action': 'require_additional_auth'},
                            {'condition': 'risk_score > 40', 'action': 'flag_for_review'}
                        ]
                    }
                }
            ]
        }
    }
]

# =============================================================================
# INSURANCE CAPABILITY TEMPLATES (Task 14.1.8)
# =============================================================================

INSURANCE_RBA_TEMPLATES = [
    {
        'name': 'Claims Processing Workflow',
        'description': 'Automated insurance claims lifecycle management',
        'category': 'claims_management',
        'industry_tags': ['Insurance'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'claim_intake',
                    'type': 'query',
                    'params': {
                        'source': 'claims_system',
                        'required_fields': ['policy_number', 'incident_date', 'claim_amount', 'supporting_documents']
                    }
                },
                {
                    'id': 'policy_validation',
                    'type': 'decision',
                    'params': {
                        'validations': [
                            'policy_active',
                            'coverage_applicable',
                            'deductible_met',
                            'claim_within_reporting_period'
                        ]
                    }
                },
                {
                    'id': 'auto_adjudication',
                    'type': 'decision',
                    'params': {
                        'auto_approve_criteria': [
                            {'condition': 'claim_amount < 5000 AND policy_in_good_standing', 'action': 'auto_approve'},
                            {'condition': 'claim_amount > 50000 OR fraud_indicators_present', 'action': 'manual_review'}
                        ]
                    }
                }
            ]
        }
    },
    {
        'name': 'Underwriting Decision Engine',
        'description': 'Automated insurance underwriting and risk assessment',
        'category': 'underwriting',
        'industry_tags': ['Insurance'],
        'workflow_definition': {
            'steps': [
                {
                    'id': 'risk_data_collection',
                    'type': 'query',
                    'params': {
                        'sources': ['credit_bureau', 'dmv_records', 'claims_history', 'property_records']
                    }
                },
                {
                    'id': 'risk_scoring',
                    'type': 'decision',
                    'params': {
                        'scoring_model': 'actuarial_risk_model_v3',
                        'factors': ['age', 'location', 'coverage_amount', 'claims_history', 'credit_score']
                    }
                },
                {
                    'id': 'underwriting_decision',
                    'type': 'decision',
                    'params': {
                        'decision_matrix': {
                            'low_risk': 'auto_approve_standard_rate',
                            'medium_risk': 'approve_with_surcharge',
                            'high_risk': 'decline_or_manual_review'
                        }
                    }
                }
            ]
        }
    }
]

# =============================================================================
# CAPABILITY REGISTRY POPULATION ENGINE
# =============================================================================

class CapabilityRegistryPopulator:
    """Populates the capability registry with baseline templates"""
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
    async def populate_registry(self) -> Dict[str, Any]:
        """Populate capability registry with all baseline templates"""
        self.logger.info("🏗️ Starting Capability Registry Population (Chapter 14.1)")
        
        results = {
            'chapter': '14.1',
            'title': 'Capability Registry Population',
            'templates_added': 0,
            'by_category': {},
            'by_industry': {},
            'errors': []
        }
        
        # All template collections
        template_collections = [
            ('SaaS RBA', SAAS_RBA_TEMPLATES),
            ('SaaS RBIA', SAAS_RBIA_TEMPLATES),
            ('SaaS AALA', SAAS_AALA_TEMPLATES),
            ('Banking RBA', BANKING_RBA_TEMPLATES),
            ('Insurance RBA', INSURANCE_RBA_TEMPLATES)
        ]
        
        try:
            pool = await self.pool_manager.get_connection()
            
            for collection_name, templates in template_collections:
                self.logger.info(f"📋 Adding {collection_name} templates ({len(templates)} templates)")
                
                for template in templates:
                    try:
                        await self._insert_capability_template(pool, template, collection_name)
                        results['templates_added'] += 1
                        
                        # Track by category and industry
                        category = template.get('category', 'unknown')
                        industry = template.get('industry_tags', ['unknown'])[0]
                        
                        results['by_category'][category] = results['by_category'].get(category, 0) + 1
                        results['by_industry'][industry] = results['by_industry'].get(industry, 0) + 1
                        
                    except Exception as template_error:
                        error_msg = f"Failed to add template {template['name']}: {template_error}"
                        self.logger.error(f"❌ {error_msg}")
                        results['errors'].append(error_msg)
            
            await pool.close()
            
            results['success'] = len(results['errors']) == 0
            self.logger.info(f"✅ Registry population completed: {results['templates_added']} templates added")
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            self.logger.error(f"❌ Registry population failed: {e}")
        
        return results
    
    async def _insert_capability_template(self, pool, template: Dict[str, Any], collection_name: str):
        """Insert a single capability template into the registry"""
        
        # Determine automation type from collection name
        if 'RBA' in collection_name:
            automation_type = 'RBA'
        elif 'RBIA' in collection_name:
            automation_type = 'RBIA'
        elif 'AALA' in collection_name:
            automation_type = 'AALA'
        else:
            automation_type = 'RBA'  # Default
        
        # Insert into dsl_capability_registry
        capability_id = str(uuid.uuid4())
        await pool.execute("""
            INSERT INTO dsl_capability_registry 
            (id, tenant_id, capability_name, capability_type, capability_definition, 
             input_schema, output_schema, created_by_user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO NOTHING
        """, 
            capability_id,
            1300,  # Default tenant for baseline templates
            template['name'],
            automation_type.lower(),
            json.dumps(template['workflow_definition']),
            json.dumps({}),  # Input schema placeholder
            json.dumps({}),  # Output schema placeholder
            1  # System user
        )
        
        # Insert into dsl_workflow_templates for drag-and-drop designer
        template_id = str(uuid.uuid4())
        await pool.execute("""
            INSERT INTO dsl_workflow_templates
            (id, tenant_id, template_name, template_description, template_definition,
             category, industry_tags, created_by_user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO NOTHING
        """,
            template_id,
            1300,  # Default tenant
            template['name'],
            template['description'],
            json.dumps(template['workflow_definition']),
            template.get('category'),
            template.get('industry_tags', []),
            1  # System user
        )
        
        self.logger.info(f"✅ Added template: {template['name']} ({automation_type})")

async def main():
    """Main execution function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize connection pool
    pool_manager = ConnectionPoolManager()
    await pool_manager.initialize()
    
    # Populate capability registry
    populator = CapabilityRegistryPopulator(pool_manager)
    results = await populator.populate_registry()
    
    # Display results
    print("\n" + "="*80)
    print("🏗️ CHAPTER 14.1 CAPABILITY REGISTRY POPULATION - RESULTS")
    print("="*80)
    print(f"📊 Overall Status: {'SUCCESS' if results.get('success', False) else 'FAILED'}")
    print(f"✅ Templates Added: {results['templates_added']}")
    
    if results['by_industry']:
        print(f"\n📋 By Industry:")
        for industry, count in results['by_industry'].items():
            print(f"   • {industry}: {count} templates")
    
    if results['by_category']:
        print(f"\n📋 By Category:")
        for category, count in results['by_category'].items():
            print(f"   • {category}: {count} templates")
    
    if results.get('errors'):
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"   • {error}")
    
    print(f"\n🎯 Registry Status:")
    print(f"   • RBA Templates: Available for SaaS, Banking, Insurance")
    print(f"   • RBIA Templates: Available for SaaS (ML-powered)")
    print(f"   • AALA Templates: Available for SaaS (AI agents)")
    
    if results.get('success', False):
        print(f"\n🎉 Capability Registry: POPULATED AND READY!")
        print(f"🎯 Next: Chapter 15.2 - Policy Pack Deployment")
    else:
        print(f"\n⚠️ Some templates failed to load. Check logs above.")
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
