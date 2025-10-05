# Banking/BFSI Industry Overlay - Chapter 9.1
# Tasks 9.1-T16: BFSI overlay (RBI/DPDP; AML/KYC; residency)

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import logging

from ..adapters.adapter_sdk import (
    PolicyAdapter, SchemaAdapter, DSLMacroAdapter, ConnectorAdapter, DashboardAdapter,
    OverlayManifest, IndustryType, OverlayType, OverlayStatus
)

logger = logging.getLogger(__name__)

class BankingPolicyAdapter:
    """Banking Policy Adapter - RBI, AML/KYC, and Basel III compliance policies"""
    
    def validate_policy(self, policy_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Banking policy against RBI and Basel III requirements"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # RBI Loan Sanction Validation
        if 'loan_amount' in policy_data:
            loan_amount = policy_data.get('loan_amount', 0)
            if loan_amount > 1000000:  # 10 Lakh threshold
                if 'credit_committee_approval' not in policy_data or not policy_data['credit_committee_approval']:
                    validation_result['issues'].append("Loan amount >10L requires credit committee approval (RBI compliance)")
                    validation_result['valid'] = False
        
        # AML/KYC Validation
        if 'customer_onboarding' in policy_data and policy_data['customer_onboarding']:
            if 'kyc_verification' not in policy_data or not policy_data['kyc_verification']:
                validation_result['issues'].append("Customer onboarding requires KYC verification (RBI AML guidelines)")
                validation_result['valid'] = False
            
            if 'aml_screening' not in policy_data or not policy_data['aml_screening']:
                validation_result['issues'].append("Customer onboarding requires AML screening (RBI compliance)")
                validation_result['valid'] = False
        
        # Data Residency Validation (DPDP)
        if 'data_processing' in policy_data and policy_data['data_processing']:
            if 'data_residency' not in policy_data or policy_data['data_residency'] != 'IN':
                validation_result['issues'].append("Customer data must be stored in India (DPDP compliance)")
                validation_result['valid'] = False
        
        # Basel III Capital Adequacy
        if 'capital_adequacy_ratio' in policy_data:
            car = policy_data.get('capital_adequacy_ratio', 0)
            if car < 9.0:  # Basel III minimum
                validation_result['issues'].append("Capital Adequacy Ratio below Basel III minimum (9%)")
                validation_result['valid'] = False
        
        return validation_result
    
    def apply_policy_overlay(self, base_policy: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Banking policy overlay to base policy"""
        merged_policy = base_policy.copy()
        
        # Add Banking-specific policy rules
        banking_rules = {
            'rbi_loan_threshold': 1000000,  # 10 Lakh
            'kyc_mandatory': True,
            'aml_screening_required': True,
            'data_residency': 'IN',
            'basel_iii_car_minimum': 9.0,
            'sanction_screening_required': True,
            'transaction_monitoring': True,
            'suspicious_activity_reporting': True,
            'audit_trail_retention_years': 7
        }
        
        merged_policy.update(banking_rules)
        merged_policy.update(overlay)
        
        return merged_policy

class BankingSchemaAdapter:
    """Banking Schema Adapter - Loan, KYC, and AML fields"""
    
    def extend_schema(self, base_schema: Dict[str, Any], extensions: Dict[str, Any]) -> Dict[str, Any]:
        """Extend base schema with Banking-specific fields"""
        extended_schema = base_schema.copy()
        
        # Add Banking-specific schema fields
        banking_fields = {
            'properties': {
                **extended_schema.get('properties', {}),
                'loan_details': {
                    'type': 'object',
                    'properties': {
                        'loan_amount': {'type': 'number', 'minimum': 0},
                        'loan_type': {'type': 'string', 'enum': ['personal', 'home', 'auto', 'business']},
                        'interest_rate': {'type': 'number', 'minimum': 0},
                        'tenure_months': {'type': 'integer', 'minimum': 1},
                        'collateral_value': {'type': 'number', 'minimum': 0},
                        'ltv_ratio': {'type': 'number', 'minimum': 0, 'maximum': 100}
                    }
                },
                'kyc_details': {
                    'type': 'object',
                    'properties': {
                        'kyc_status': {'type': 'string', 'enum': ['pending', 'verified', 'rejected']},
                        'kyc_type': {'type': 'string', 'enum': ['simplified', 'basic', 'enhanced']},
                        'document_verification': {'type': 'boolean'},
                        'biometric_verification': {'type': 'boolean'},
                        'risk_category': {'type': 'string', 'enum': ['low', 'medium', 'high']},
                        'kyc_expiry_date': {'type': 'string', 'format': 'date'}
                    }
                },
                'aml_screening': {
                    'type': 'object',
                    'properties': {
                        'pep_check': {'type': 'boolean'},
                        'sanction_screening': {'type': 'boolean'},
                        'adverse_media_check': {'type': 'boolean'},
                        'risk_score': {'type': 'number', 'minimum': 0, 'maximum': 100},
                        'screening_date': {'type': 'string', 'format': 'date-time'},
                        'screening_provider': {'type': 'string'}
                    }
                },
                'transaction_monitoring': {
                    'type': 'object',
                    'properties': {
                        'transaction_amount': {'type': 'number', 'minimum': 0},
                        'transaction_type': {'type': 'string'},
                        'suspicious_flag': {'type': 'boolean'},
                        'monitoring_rules_triggered': {'type': 'array', 'items': {'type': 'string'}},
                        'investigation_status': {'type': 'string', 'enum': ['none', 'pending', 'completed']}
                    }
                },
                'regulatory_reporting': {
                    'type': 'object',
                    'properties': {
                        'ctr_required': {'type': 'boolean'},
                        'str_filed': {'type': 'boolean'},
                        'rbi_reporting': {'type': 'boolean'},
                        'compliance_status': {'type': 'string', 'enum': ['compliant', 'non_compliant', 'under_review']}
                    }
                },
                'capital_adequacy': {
                    'type': 'object',
                    'properties': {
                        'car_ratio': {'type': 'number', 'minimum': 0},
                        'tier1_capital': {'type': 'number', 'minimum': 0},
                        'tier2_capital': {'type': 'number', 'minimum': 0},
                        'risk_weighted_assets': {'type': 'number', 'minimum': 0},
                        'basel_iii_compliant': {'type': 'boolean'}
                    }
                }
            }
        }
        
        extended_schema.update(banking_fields)
        extended_schema.update(extensions)
        
        return extended_schema
    
    def validate_schema_compatibility(self, base_schema: Dict[str, Any], overlay_schema: Dict[str, Any]) -> bool:
        """Validate Banking schema compatibility"""
        # Check for required Banking fields
        required_fields = ['kyc_details', 'aml_screening', 'regulatory_reporting']
        overlay_properties = overlay_schema.get('properties', {})
        
        for field in required_fields:
            if field not in overlay_properties:
                logger.warning(f"⚠️ Banking overlay missing recommended field: {field}")
        
        return True

class BankingDSLMacroAdapter:
    """Banking DSL Macro Adapter - AML/KYC and loan processing workflows"""
    
    def __init__(self):
        self.macros = {
            'kyc_verification_flow': {
                'description': 'Complete KYC verification workflow',
                'parameters': ['customer_id', 'kyc_type', 'risk_category'],
                'template': {
                    'type': 'agent_call',
                    'params': {
                        'agent': 'kyc_verification_agent',
                        'customer_id': '${customer_id}',
                        'kyc_type': '${kyc_type}',
                        'risk_category': '${risk_category}',
                        'compliance_framework': 'RBI_KYC'
                    }
                }
            },
            'aml_screening_check': {
                'description': 'Perform AML screening and sanction checks',
                'parameters': ['customer_name', 'screening_level'],
                'template': {
                    'type': 'query',
                    'params': {
                        'source': 'aml_database',
                        'query': 'SELECT * FROM sanction_lists WHERE name LIKE ${customer_name}',
                        'screening_level': '${screening_level}'
                    }
                }
            },
            'loan_sanction_workflow': {
                'description': 'Loan sanction and approval workflow',
                'parameters': ['loan_amount', 'customer_id', 'loan_type'],
                'template': {
                    'type': 'decision',
                    'params': {
                        'condition': 'loan_amount > 1000000',
                        'true_action': {
                            'type': 'governance',
                            'params': {
                                'policy_id': 'rbi_loan_approval',
                                'approval_required': 'credit_committee'
                            }
                        },
                        'false_action': {
                            'type': 'governance',
                            'params': {
                                'policy_id': 'standard_loan_approval',
                                'approval_required': 'branch_manager'
                            }
                        }
                    }
                }
            },
            'transaction_monitoring': {
                'description': 'Monitor transactions for suspicious activity',
                'parameters': ['transaction_amount', 'customer_risk_score'],
                'template': {
                    'type': 'ml_decision',
                    'params': {
                        'model': 'transaction_anomaly_detector',
                        'features': ['transaction_amount', 'customer_risk_score', 'transaction_pattern'],
                        'threshold': 0.7
                    }
                }
            },
            'regulatory_reporting': {
                'description': 'Generate regulatory reports for RBI',
                'parameters': ['report_type', 'reporting_period'],
                'template': {
                    'type': 'query',
                    'params': {
                        'source': 'core_banking',
                        'query': 'EXEC generate_rbi_report @report_type=${report_type}, @period=${reporting_period}',
                        'output_format': 'xml'
                    }
                }
            }
        }
    
    def register_macros(self, macros: Dict[str, Any]) -> bool:
        """Register Banking-specific DSL macros"""
        try:
            self.macros.update(macros)
            logger.info(f"✅ Registered {len(macros)} Banking DSL macros")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to register Banking macros: {e}")
            return False
    
    def expand_macro(self, macro_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Expand Banking DSL macro with parameters"""
        if macro_name not in self.macros:
            raise ValueError(f"Unknown Banking macro: {macro_name}")
        
        macro = self.macros[macro_name]
        template = macro['template'].copy()
        
        # Replace parameter placeholders
        template_str = json.dumps(template)
        for param, value in parameters.items():
            template_str = template_str.replace(f"${{{param}}}", str(value))
        
        return json.loads(template_str)

class BankingConnectorAdapter:
    """Banking Connector Adapter - Core banking, AML, and regulatory integrations"""
    
    def create_connector_shim(self, connector_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Banking-normalized connector shim"""
        
        connector_configs = {
            'core_banking': {
                'type': 'core_banking_system',
                'auth_type': 'certificate',
                'endpoints': {
                    'accounts': '/api/v1/accounts',
                    'transactions': '/api/v1/transactions',
                    'loans': '/api/v1/loans',
                    'customers': '/api/v1/customers'
                },
                'rate_limits': {
                    'requests_per_minute': 500
                },
                'retry_config': {
                    'max_retries': 5,
                    'backoff_factor': 2
                },
                'security': {
                    'encryption': 'AES-256',
                    'message_signing': True
                }
            },
            'aml_system': {
                'type': 'aml_screening',
                'auth_type': 'api_key',
                'endpoints': {
                    'screening': '/api/v2/screening',
                    'watchlists': '/api/v2/watchlists',
                    'sanctions': '/api/v2/sanctions'
                },
                'rate_limits': {
                    'requests_per_minute': 100
                },
                'retry_config': {
                    'max_retries': 3,
                    'backoff_factor': 1.5
                }
            },
            'rbi_reporting': {
                'type': 'regulatory_reporting',
                'auth_type': 'digital_certificate',
                'endpoints': {
                    'submit_report': '/rbi/api/submit',
                    'report_status': '/rbi/api/status',
                    'acknowledgment': '/rbi/api/ack'
                },
                'rate_limits': {
                    'requests_per_hour': 50
                },
                'retry_config': {
                    'max_retries': 5,
                    'backoff_factor': 3
                },
                'security': {
                    'digital_signature': True,
                    'encryption': 'RSA-2048'
                }
            },
            'kyc_system': {
                'type': 'kyc_verification',
                'auth_type': 'oauth2',
                'endpoints': {
                    'verify_documents': '/kyc/v1/verify',
                    'biometric_match': '/kyc/v1/biometric',
                    'address_verification': '/kyc/v1/address'
                },
                'rate_limits': {
                    'requests_per_minute': 200
                },
                'retry_config': {
                    'max_retries': 3,
                    'backoff_factor': 2
                }
            }
        }
        
        if connector_type not in connector_configs:
            raise ValueError(f"Unsupported Banking connector type: {connector_type}")
        
        base_config = connector_configs[connector_type].copy()
        base_config.update(config)
        
        return base_config
    
    def validate_connector_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Banking connector configuration"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        required_fields = ['type', 'auth_type', 'endpoints']
        for field in required_fields:
            if field not in config:
                validation_result['issues'].append(f"Missing required field: {field}")
                validation_result['valid'] = False
        
        # Validate security requirements for Banking
        if 'security' not in config:
            validation_result['warnings'].append("No security configuration - Banking requires encryption and signing")
        
        # Validate rate limits for regulatory compliance
        if 'rate_limits' not in config:
            validation_result['warnings'].append("No rate limits specified - may violate regulatory API guidelines")
        
        return validation_result

class BankingDashboardAdapter:
    """Banking Dashboard Adapter - Loan portfolio, AML, and regulatory KPIs"""
    
    def __init__(self):
        self.industry_kpis = [
            'capital_adequacy_ratio',
            'non_performing_assets_ratio',
            'loan_to_deposit_ratio',
            'return_on_assets',
            'return_on_equity',
            'net_interest_margin',
            'cost_to_income_ratio',
            'credit_growth_rate',
            'deposit_growth_rate',
            'kyc_completion_rate',
            'aml_screening_coverage',
            'suspicious_transaction_rate',
            'regulatory_compliance_score',
            'loan_approval_time',
            'customer_onboarding_time'
        ]
    
    def create_dashboard_config(self, dashboard_type: str, kpis: List[str]) -> Dict[str, Any]:
        """Create Banking-specific dashboard configuration"""
        
        dashboard_configs = {
            'regulatory_dashboard': {
                'title': 'Banking Regulatory Dashboard',
                'description': 'Monitor RBI compliance and regulatory metrics',
                'layout': 'grid',
                'refresh_interval': 300,  # 5 minutes
                'widgets': [
                    {
                        'type': 'metric_card',
                        'title': 'Capital Adequacy Ratio',
                        'kpi': 'capital_adequacy_ratio',
                        'format': 'percentage',
                        'size': 'large',
                        'threshold': {'min': 9.0, 'target': 12.0}
                    },
                    {
                        'type': 'metric_card',
                        'title': 'NPA Ratio',
                        'kpi': 'non_performing_assets_ratio',
                        'format': 'percentage',
                        'size': 'large',
                        'threshold': {'max': 4.0, 'target': 2.0}
                    },
                    {
                        'type': 'chart',
                        'title': 'Regulatory Compliance Trend',
                        'chart_type': 'line',
                        'kpis': ['regulatory_compliance_score'],
                        'time_range': '12M'
                    }
                ]
            },
            'aml_dashboard': {
                'title': 'AML Compliance Dashboard',
                'description': 'Monitor AML screening and suspicious activity',
                'layout': 'grid',
                'refresh_interval': 180,  # 3 minutes
                'widgets': [
                    {
                        'type': 'metric_card',
                        'title': 'KYC Completion Rate',
                        'kpi': 'kyc_completion_rate',
                        'format': 'percentage',
                        'size': 'medium'
                    },
                    {
                        'type': 'metric_card',
                        'title': 'AML Screening Coverage',
                        'kpi': 'aml_screening_coverage',
                        'format': 'percentage',
                        'size': 'medium'
                    },
                    {
                        'type': 'alert_list',
                        'title': 'Suspicious Transactions',
                        'kpi': 'suspicious_transaction_rate',
                        'alert_threshold': 0.05
                    }
                ]
            },
            'loan_portfolio_dashboard': {
                'title': 'Loan Portfolio Dashboard',
                'description': 'Track loan performance and credit metrics',
                'layout': 'grid',
                'refresh_interval': 900,  # 15 minutes
                'widgets': [
                    {
                        'type': 'metric_card',
                        'title': 'Credit Growth Rate',
                        'kpi': 'credit_growth_rate',
                        'format': 'percentage',
                        'size': 'large'
                    },
                    {
                        'type': 'metric_card',
                        'title': 'Loan to Deposit Ratio',
                        'kpi': 'loan_to_deposit_ratio',
                        'format': 'percentage',
                        'size': 'medium'
                    },
                    {
                        'type': 'pie_chart',
                        'title': 'Loan Portfolio Mix',
                        'dimensions': ['loan_type'],
                        'metric': 'loan_amount'
                    }
                ]
            }
        }
        
        if dashboard_type not in dashboard_configs:
            raise ValueError(f"Unsupported Banking dashboard type: {dashboard_type}")
        
        config = dashboard_configs[dashboard_type].copy()
        
        # Filter widgets based on requested KPIs
        if kpis:
            filtered_widgets = []
            for widget in config['widgets']:
                widget_kpi = widget.get('kpi')
                widget_kpis = widget.get('kpis', [])
                
                if widget_kpi in kpis or any(k in kpis for k in widget_kpis):
                    filtered_widgets.append(widget)
            
            config['widgets'] = filtered_widgets
        
        return config
    
    def get_industry_kpis(self) -> List[str]:
        """Get Banking industry-specific KPIs"""
        return self.industry_kpis.copy()

def create_banking_overlay() -> OverlayManifest:
    """Create Banking industry overlay manifest"""
    
    overlay = OverlayManifest(
        overlay_id="banking_bfsi_v1",
        name="Banking BFSI Overlay",
        version="1.0.0",
        industry=IndustryType.BANKING,
        overlay_type=OverlayType.COMPLIANCE,
        description="Banking BFSI overlay with RBI, AML/KYC, and Basel III compliance for loan processing and regulatory reporting",
        created_by="system",
        created_at=datetime.now(timezone.utc).isoformat(),
        compliance_frameworks=["RBI", "BASEL_III", "DPDP", "AML"],
        residency_requirements=["IN"],
        sod_requirements=["maker_checker", "credit_committee_approval"],
        schema_extensions={
            'banking_loan_fields': {
                'loan_amount': 'number',
                'kyc_status': 'string',
                'aml_screening': 'object'
            }
        },
        policy_rules={
            'rbi_loan_threshold': 1000000,
            'kyc_mandatory': True,
            'data_residency': 'IN',
            'basel_iii_car_minimum': 9.0
        },
        dsl_macros={
            'kyc_verification_flow': 'Complete KYC verification workflow',
            'aml_screening_check': 'Perform AML and sanction screening',
            'loan_sanction_workflow': 'Loan sanction and approval process'
        },
        connector_configs={
            'core_banking': 'Core banking system integration',
            'aml_system': 'AML screening system',
            'rbi_reporting': 'RBI regulatory reporting'
        },
        status=OverlayStatus.PUBLISHED
    )
    
    return overlay

# Initialize Banking adapters
banking_policy_adapter = BankingPolicyAdapter()
banking_schema_adapter = BankingSchemaAdapter()
banking_dsl_macro_adapter = BankingDSLMacroAdapter()
banking_connector_adapter = BankingConnectorAdapter()
banking_dashboard_adapter = BankingDashboardAdapter()
