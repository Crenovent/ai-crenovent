"""
Company Profile Analyzer
Analyzes CSV data to detect company characteristics and hierarchy patterns.

Features:
- Company size detection and classification
- Hierarchy model detection (geographic, segment, hybrid, custom)
- Custom terminology extraction
- Dynamic schema generation
- SaaS revenue function classification
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class CompanyProfileAnalyzer:
    """
    Analyzes company characteristics from CSV data to enable adaptive hierarchy assignment
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # SaaS Revenue Function Patterns
        self.saas_function_patterns = {
            'sales': {
                'titles': ['account executive', 'ae', 'sales executive', 'sdr', 'bdr', 'sales development', 
                          'business development', 'sales manager', 'sales director', 'vp sales', 'cro'],
                'levels': ['c_level', 'vp_level', 'director', 'manager', 'senior', 'junior'],
                'segments': ['enterprise', 'mid_market', 'smb', 'strategic']
            },
            'marketing': {
                'titles': ['demand generation', 'demand gen', 'growth marketing', 'product marketing', 
                          'pmm', 'content marketing', 'digital marketing', 'cmo', 'marketing manager'],
                'levels': ['c_level', 'vp_level', 'director', 'manager', 'senior', 'specialist'],
                'segments': ['enterprise', 'growth', 'retention', 'acquisition']
            },
            'customer_success': {
                'titles': ['customer success', 'csm', 'implementation', 'onboarding', 'support', 
                          'technical support', 'customer experience', 'vp customer success'],
                'levels': ['c_level', 'vp_level', 'director', 'manager', 'senior', 'specialist'],
                'segments': ['enterprise', 'commercial', 'smb', 'self_service']
            },
            'revenue_operations': {
                'titles': ['revenue operations', 'revops', 'sales operations', 'revenue analyst', 
                          'sales analyst', 'business analyst', 'operations manager'],
                'levels': ['c_level', 'vp_level', 'director', 'manager', 'senior', 'analyst'],
                'segments': ['cross_functional', 'enterprise', 'commercial']
            },
            'partnerships': {
                'titles': ['channel manager', 'partner manager', 'alliance manager', 'strategic partnerships',
                          'business development', 'bd manager', 'partnerships'],
                'levels': ['vp_level', 'director', 'manager', 'senior'],
                'segments': ['strategic', 'channel', 'reseller']
            },
            'product': {
                'titles': ['product marketing', 'pmm', 'growth product', 'product growth', 'product manager',
                          'pm', 'product director', 'cpo', 'head of product'],
                'levels': ['c_level', 'vp_level', 'director', 'manager', 'senior'],
                'segments': ['enterprise', 'growth', 'platform', 'core']
            }
        }
        
        # Geographic indicators
        self.geographic_patterns = {
            'global': ['global', 'worldwide', 'international'],
            'regional': ['americas', 'emea', 'apac', 'latam', 'north america', 'europe', 'asia pacific'],
            'country': ['us', 'usa', 'uk', 'canada', 'germany', 'france', 'australia', 'india'],
            'territory': ['west', 'east', 'central', 'north', 'south', 'northeast', 'southwest']
        }
        
        # Level indicators
        self.level_patterns = {
            'c_level': ['ceo', 'cto', 'cfo', 'cmo', 'cro', 'cpo', 'chief'],
            'vp_level': ['vp', 'vice president', 'head of', 'svp', 'senior vice president'],
            'director': ['director', 'senior director', 'principal'],
            'manager': ['manager', 'team lead', 'lead', 'supervisor'],
            'senior': ['senior', 'sr', 'principal', 'staff'],
            'junior': ['junior', 'jr', 'associate', 'coordinator', 'specialist']
        }

    def analyze_company_profile(self, users_data: List[Dict], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze company characteristics from user data
        """
        try:
            self.logger.info("🔍 Analyzing company profile from CSV data...")
            
            # Basic company metrics
            employee_count = len(users_data)
            
            # Analyze job titles and functions
            function_analysis = self._analyze_revenue_functions(users_data)
            
            # Detect hierarchy model
            hierarchy_analysis = self._detect_hierarchy_model(users_data)
            
            # Extract custom terminology
            terminology_analysis = self._extract_custom_terminology(users_data)
            
            # Generate dynamic schema
            dynamic_schema = self._generate_dynamic_schema(hierarchy_analysis, terminology_analysis)
            
            # Calculate confidence score
            confidence_score = self._calculate_profile_confidence(
                function_analysis, hierarchy_analysis, terminology_analysis
            )
            
            profile = {
                'employee_count': employee_count,
                'company_size_category': self._categorize_company_size(employee_count),
                'detected_hierarchy_model': hierarchy_analysis['model_type'],
                'hierarchy_depth': hierarchy_analysis['depth'],
                'geographic_levels': hierarchy_analysis['geographic_levels'],
                'functional_levels': hierarchy_analysis['functional_levels'],
                'custom_terminology': terminology_analysis,
                'revenue_function_distribution': function_analysis,
                'dynamic_schema': dynamic_schema,
                'confidence_score': confidence_score,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"✅ Company profile analysis complete: {profile['company_size_category']} company with {profile['detected_hierarchy_model']} model")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"❌ Company profile analysis failed: {e}")
            return self._get_default_profile(len(users_data))

    def _analyze_revenue_functions(self, users_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze distribution of revenue functions in the company
        """
        function_counts = {}
        total_users = len(users_data)
        
        for user in users_data:
            job_title = str(user.get('job_title', '')).lower()
            department = str(user.get('department', '')).lower()
            
            # Classify function based on title and department
            classified_function = self._classify_revenue_function(job_title, department)
            function_counts[classified_function] = function_counts.get(classified_function, 0) + 1
        
        # Calculate percentages
        function_distribution = {}
        for function, count in function_counts.items():
            function_distribution[function] = {
                'count': count,
                'percentage': round((count / total_users) * 100, 2)
            }
        
        return function_distribution

    def _classify_revenue_function(self, job_title: str, department: str) -> str:
        """
        Classify a user into a revenue function category
        """
        combined_text = f"{job_title} {department}".lower()
        
        # Check each function pattern
        for function, patterns in self.saas_function_patterns.items():
            for title_pattern in patterns['titles']:
                if title_pattern in combined_text:
                    return function
        
        # Default classification
        return 'other'

    def _detect_hierarchy_model(self, users_data: List[Dict]) -> Dict[str, Any]:
        """
        Detect the type of hierarchy model used by the company
        """
        geographic_indicators = 0
        segment_indicators = 0
        functional_indicators = 0
        
        # Analyze existing fields for hierarchy patterns
        for user in users_data:
            # Check for geographic patterns
            for field in ['region', 'area', 'territory', 'location', 'office']:
                value = str(user.get(field, '')).lower()
                if value and any(pattern in value for patterns in self.geographic_patterns.values() for pattern in patterns):
                    geographic_indicators += 1
                    break
            
            # Check for segment patterns
            segment_value = str(user.get('segment', '')).lower()
            if segment_value and any(seg in segment_value for seg in ['enterprise', 'mid market', 'smb', 'commercial']):
                segment_indicators += 1
            
            # Check for functional patterns
            job_title = str(user.get('job_title', '')).lower()
            if any(level in job_title for levels in self.level_patterns.values() for level in levels):
                functional_indicators += 1
        
        total_users = len(users_data)
        geographic_ratio = geographic_indicators / total_users if total_users > 0 else 0
        segment_ratio = segment_indicators / total_users if total_users > 0 else 0
        functional_ratio = functional_indicators / total_users if total_users > 0 else 0
        
        # Determine model type
        if geographic_ratio > 0.3 and segment_ratio > 0.3:
            model_type = 'hybrid'
        elif geographic_ratio > 0.3:
            model_type = 'geographic'
        elif segment_ratio > 0.3:
            model_type = 'segment'
        else:
            model_type = 'custom'
        
        # Detect hierarchy depth
        depth = self._detect_hierarchy_depth(users_data)
        
        # Generate level mappings
        geographic_levels = self._generate_geographic_levels(depth, model_type)
        functional_levels = self._generate_functional_levels(depth, model_type)
        
        return {
            'model_type': model_type,
            'depth': depth,
            'geographic_levels': geographic_levels,
            'functional_levels': functional_levels,
            'indicators': {
                'geographic_ratio': geographic_ratio,
                'segment_ratio': segment_ratio,
                'functional_ratio': functional_ratio
            }
        }

    def _detect_hierarchy_depth(self, users_data: List[Dict]) -> int:
        """
        Detect the depth of organizational hierarchy
        """
        # Count unique management levels
        levels_found = set()
        
        for user in users_data:
            job_title = str(user.get('job_title', '')).lower()
            
            # Check for level indicators
            for level, patterns in self.level_patterns.items():
                if any(pattern in job_title for pattern in patterns):
                    levels_found.add(level)
        
        # Estimate depth based on company size and levels found
        employee_count = len(users_data)
        base_depth = len(levels_found) if levels_found else 2
        
        # Adjust based on company size
        if employee_count < 50:
            return max(2, min(base_depth, 3))
        elif employee_count < 200:
            return max(3, min(base_depth, 4))
        elif employee_count < 1000:
            return max(4, min(base_depth, 5))
        else:
            return max(5, min(base_depth, 7))

    def _generate_geographic_levels(self, depth: int, model_type: str) -> Dict[str, str]:
        """
        Generate geographic level mappings based on detected depth
        """
        if model_type in ['geographic', 'hybrid']:
            if depth <= 3:
                return {
                    'level_1': 'Region',
                    'level_2': 'Area',
                    'level_3': 'Territory'
                }
            elif depth == 4:
                return {
                    'level_1': 'Global Region',
                    'level_2': 'Country',
                    'level_3': 'Area',
                    'level_4': 'Territory'
                }
            else:
                return {
                    'level_1': 'Global Region',
                    'level_2': 'Country',
                    'level_3': 'State/Province',
                    'level_4': 'Area',
                    'level_5': 'District',
                    'level_6': 'Territory'
                }
        else:
            return {'level_1': 'Region'}

    def _generate_functional_levels(self, depth: int, model_type: str) -> Dict[str, str]:
        """
        Generate functional level mappings based on detected depth
        """
        if depth <= 3:
            return {
                'level_1': 'Division',
                'level_2': 'Department',
                'level_3': 'Team'
            }
        elif depth == 4:
            return {
                'level_1': 'Business Unit',
                'level_2': 'Division',
                'level_3': 'Department',
                'level_4': 'Team'
            }
        else:
            return {
                'level_1': 'Organization',
                'level_2': 'Business Unit',
                'level_3': 'Division',
                'level_4': 'Department',
                'level_5': 'Function',
                'level_6': 'Team'
            }

    def _extract_custom_terminology(self, users_data: List[Dict]) -> Dict[str, str]:
        """
        Extract company-specific terminology from existing data
        """
        terminology = {}
        
        # Analyze field names and values for custom terms
        if users_data:
            sample_user = users_data[0]
            
            # Check for custom field names
            for field_name in sample_user.keys():
                field_lower = field_name.lower()
                
                # Map common variations
                if 'region' in field_lower and field_name != 'region':
                    terminology['region'] = field_name.title()
                elif 'territory' in field_lower and field_name != 'territory':
                    terminology['territory'] = field_name.title()
                elif 'area' in field_lower and field_name != 'area':
                    terminology['area'] = field_name.title()
                elif 'segment' in field_lower and field_name != 'segment':
                    terminology['segment'] = field_name.title()
        
        return terminology

    def _generate_dynamic_schema(self, hierarchy_analysis: Dict, terminology_analysis: Dict) -> Dict[str, Any]:
        """
        Generate dynamic field schema based on analysis
        """
        schema = {
            'geographic_fields': [],
            'functional_fields': [],
            'required_fields': ['email', 'name', 'job_title'],
            'optional_fields': []
        }
        
        # Add geographic fields based on detected levels
        for level_key, level_name in hierarchy_analysis['geographic_levels'].items():
            field_name = level_name.lower().replace(' ', '_')
            schema['geographic_fields'].append({
                'field_name': field_name,
                'display_name': terminology_analysis.get(field_name, level_name),
                'level': level_key,
                'required': level_key in ['level_1', 'level_2']
            })
        
        # Add functional fields
        for level_key, level_name in hierarchy_analysis['functional_levels'].items():
            field_name = level_name.lower().replace(' ', '_')
            schema['functional_fields'].append({
                'field_name': field_name,
                'display_name': terminology_analysis.get(field_name, level_name),
                'level': level_key,
                'required': level_key in ['level_1']
            })
        
        return schema

    def _calculate_profile_confidence(self, function_analysis: Dict, hierarchy_analysis: Dict, terminology_analysis: Dict) -> float:
        """
        Calculate confidence score for the company profile analysis
        """
        confidence_factors = []
        
        # Function analysis confidence
        total_classified = sum(data['count'] for func, data in function_analysis.items() if func != 'other')
        total_users = sum(data['count'] for data in function_analysis.values())
        if total_users > 0:
            function_confidence = total_classified / total_users
            confidence_factors.append(function_confidence * 0.4)
        
        # Hierarchy model confidence
        indicators = hierarchy_analysis['indicators']
        model_confidence = max(indicators['geographic_ratio'], indicators['segment_ratio'], indicators['functional_ratio'])
        confidence_factors.append(model_confidence * 0.4)
        
        # Data completeness confidence
        completeness_confidence = min(1.0, len(terminology_analysis) / 4)  # Expect up to 4 custom terms
        confidence_factors.append(completeness_confidence * 0.2)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) if confidence_factors else 0.5
        return round(min(1.0, max(0.1, overall_confidence)), 2)

    def _categorize_company_size(self, employee_count: int) -> str:
        """
        Categorize company size based on employee count
        """
        if employee_count < 50:
            return 'startup'
        elif employee_count < 200:
            return 'small'
        elif employee_count < 1000:
            return 'medium'
        elif employee_count < 5000:
            return 'large'
        else:
            return 'enterprise'

    def _get_default_profile(self, employee_count: int) -> Dict[str, Any]:
        """
        Return default profile when analysis fails
        """
        return {
            'employee_count': employee_count,
            'company_size_category': self._categorize_company_size(employee_count),
            'detected_hierarchy_model': 'custom',
            'hierarchy_depth': 3,
            'geographic_levels': {'level_1': 'Region', 'level_2': 'Territory'},
            'functional_levels': {'level_1': 'Department', 'level_2': 'Team'},
            'custom_terminology': {},
            'revenue_function_distribution': {'other': {'count': employee_count, 'percentage': 100.0}},
            'dynamic_schema': {
                'geographic_fields': [{'field_name': 'region', 'display_name': 'Region', 'level': 'level_1', 'required': True}],
                'functional_fields': [{'field_name': 'department', 'display_name': 'Department', 'level': 'level_1', 'required': True}],
                'required_fields': ['email', 'name', 'job_title'],
                'optional_fields': []
            },
            'confidence_score': 0.3,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
