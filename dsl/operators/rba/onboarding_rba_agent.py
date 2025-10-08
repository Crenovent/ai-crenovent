"""
Onboarding RBA Agent
Single-purpose RBA agent for user hierarchy field assignment during CSV onboarding.

This agent ONLY handles dynamic field assignment for missing hierarchy fields:
- Region assignment based on role/title analysis
- Segment classification using role-based distribution
- Module access based on organizational level
- Territory assignment with geographic distribution
- Level inference from title keywords

Lightweight, modular, configuration-driven following RevAI Pro architecture.
"""

import logging
import hashlib
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent
from .company_profile_analyzer import CompanyProfileAnalyzer
from .confidence_scoring_engine import ConfidenceScoringEngine
from .ultra_dynamic_geolocation_service import UltraDynamicGeolocationService
from .dynamic_field_mapper import dynamic_field_mapper
from .centralized_module_assignment import centralized_module_assignment

logger = logging.getLogger(__name__)

class OnboardingRBAAgent(EnhancedBaseRBAAgent):
    """
    RBA Agent for dynamic user onboarding and hierarchy field assignment
    
    Features:
    - Smart region assignment (CEO/VP geographic logic + hash distribution)
    - Role-based segment classification with realistic distributions
    - Dynamic module access based on organizational level
    - Territory assignment with variety and geographic hints
    - Level inference from title keywords and patterns
    """
    
    AGENT_NAME = "onboarding_field_assignment"
    SUPPORTED_ANALYSIS_TYPES = ["onboarding_field_assignment", "hierarchy_field_assignment", "user_onboarding"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Smart Location Mapper will be initialized when needed
        self.location_mapper = None
        
        # Initialize new learning components
        self.company_analyzer = CompanyProfileAnalyzer(config)
        self.confidence_engine = ConfidenceScoringEngine(config)
        self.geolocation_service = UltraDynamicGeolocationService()
        
        # Configuration for field assignment logic
        self.region_weights = {
            'north_america': 0.45,
            'emea': 0.25,
            'apac': 0.20,
            'latam': 0.10
        }
        
        self.segment_distributions = {
            'c_level': {'enterprise': 1.0, 'mid_market': 0.0, 'smb': 0.0},
            'vp_level': {'enterprise': 0.8, 'mid_market': 0.2, 'smb': 0.0},
            'director': {'enterprise': 0.6, 'mid_market': 0.3, 'smb': 0.1},
            'senior_manager': {'enterprise': 0.4, 'mid_market': 0.4, 'smb': 0.2},
            'manager': {'enterprise': 0.3, 'mid_market': 0.4, 'smb': 0.3},
            'senior_ic': {'enterprise': 0.2, 'mid_market': 0.5, 'smb': 0.3},
            'ic': {'enterprise': 0.1, 'mid_market': 0.3, 'smb': 0.6}
        }
        
        # Module assignment now handled by centralized_module_assignment
        # Old module_access_by_level removed - logic moved to centralized system
    
    def _define_config_schema(self) -> Dict[str, Any]:
        """Define configuration schema for onboarding agent"""
        return {
            'enable_region_assignment': {
                'type': 'bool',
                'default': True,
                'description': 'Enable automatic region assignment',
                'ui_component': 'checkbox'
            },
            'enable_segment_assignment': {
                'type': 'bool', 
                'default': True,
                'description': 'Enable segment classification',
                'ui_component': 'checkbox'
            },
            'enable_module_assignment': {
                'type': 'bool',
                'default': True,
                'description': 'Enable module access assignment',
                'ui_component': 'checkbox'
            },
            'enable_level_inference': {
                'type': 'bool',
                'default': True,
                'description': 'Enable level inference from titles',
                'ui_component': 'checkbox'
            },
            'preserve_existing_data': {
                'type': 'bool',
                'default': True,
                'description': 'Only fill empty fields, preserve existing data',
                'ui_component': 'checkbox'
            },
            'region_distribution_strategy': {
                'type': 'enum',
                'options': ['geographic_hints', 'hash_based', 'weighted_random', 'smart_location'],
                'default': 'smart_location',
                'description': 'Strategy for region assignment',
                'ui_component': 'dropdown'
            },
            'enable_smart_location_mapping': {
                'type': 'bool',
                'default': True,
                'description': 'Enable intelligent location detection and mapping',
                'ui_component': 'checkbox'
            },
            'create_hierarchical_structure': {
                'type': 'bool',
                'default': True,
                'description': 'Create hierarchical geographic structure (Region > Area > District > Territory)',
                'ui_component': 'checkbox'
            },
            'assign_heads_by_geography': {
                'type': 'bool',
                'default': True,
                'description': 'Assign heads/managers based on geographic proximity',
                'ui_component': 'checkbox'
            }
        }
    
    def _define_result_schema(self) -> Dict[str, Any]:
        """Define result schema for onboarding results"""
        return {
            'processed_users': {
                'type': 'array',
                'description': 'List of processed users with assigned fields'
            },
            'assignment_statistics': {
                'type': 'object',
                'description': 'Statistics about field assignments'
            },
            'validation_errors': {
                'type': 'array', 
                'description': 'List of validation errors encountered'
            },
            'confidence_scores': {
                'type': 'object',
                'description': 'Confidence scores for each assignment type'
            },
            'geographic_hierarchy': {
                'type': 'object',
                'description': 'Generated geographic hierarchy structure'
            },
            'location_intelligence': {
                'type': 'object',
                'description': 'Smart location mapping results'
            }
        }
    
    async def _execute_rba_logic(self, context: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute onboarding field assignment logic"""
        try:
            # Debug: Check what's being passed to RBA logic
            import os
            debug_path = os.path.join(os.getcwd(), 'debug_location_intelligence.txt')
            with open(debug_path, 'a') as f:
                f.write(f"=== OnboardingRBAAgent._execute_rba_logic CALLED ===\n")
                f.write(f"Context keys: {list(context.keys())}\n")
                f.write(f"Config keys: {list(config.keys())}\n")
                f.write("=" * 50 + "\n")
            
            users_data = context.get('users_data', [])
            workflow_config = context.get('workflow_config', {})
            
            # Add users_data to workflow_config for hierarchy analysis
            workflow_config['users_data'] = users_data
            
            # Step 0: Initialize dynamic field mappings
            if users_data:
                self.field_mappings = self._initialize_field_mappings(users_data)
                self.logger.info(f"🔄 Dynamic field mappings initialized: {len(self.field_mappings)} fields mapped")
            else:
                self.field_mappings = {}
            
            if not users_data:
                return {
                    'success': False,
                    'error': 'No user data provided for processing',
                    'processed_users': [],
                    'assignment_statistics': {}
                }
            
            self.logger.info(f"🎯 Processing {len(users_data)} users for field assignment")
            
            # Step 0: Analyze Company Profile
            self.logger.info("🏢 Analyzing company profile and hierarchy patterns...")
            company_profile = self.company_analyzer.analyze_company_profile(users_data, context)
            self._current_company_profile = company_profile  # Store for confidence calculation
            self.logger.info(f"✅ Company profile analysis complete: {company_profile['company_size_category']} company with {company_profile['detected_hierarchy_model']} hierarchy model (confidence: {company_profile['confidence_score']})")
            
            # Step 1: Execute Smart Location Mapping (if enabled)
            location_intelligence = {}
            geographic_hierarchy = {}
            
            # Debug: Check if smart location mapping is enabled
            import os
            debug_path = os.path.join(os.getcwd(), 'debug_location_intelligence.txt')
            with open(debug_path, 'a') as f:
                f.write(f"=== OnboardingRBAAgent location mapping check ===\n")
                f.write(f"enable_smart_location_mapping: {config.get('enable_smart_location_mapping')}\n")
                f.write(f"Config keys: {list(config.keys())}\n")
            
            if config.get('enable_smart_location_mapping', True):
                with open(debug_path, 'a') as f:
                    f.write(f"Smart location mapping is ENABLED, executing...\n")
                
                self.logger.info("🗺️ Executing Smart Location Mapping...")
                try:
                    # Lazy import to avoid circular dependency
                    if self.location_mapper is None:
                        from .smart_location_mapper import SmartLocationMapperRBAAgent
                        self.location_mapper = SmartLocationMapperRBAAgent(config)
                    
                    with open(debug_path, 'a') as f:
                        f.write(f"Calling SmartLocationMapperRBAAgent._execute_rba_logic\n")
                    
                    location_result = await self.location_mapper._execute_rba_logic(context, config)
                    
                    with open(debug_path, 'a') as f:
                        f.write(f"SmartLocationMapperRBAAgent result: {location_result.get('success', False)}\n")
                    
                    if location_result.get('success'):
                        location_intelligence = location_result
                        geographic_hierarchy = location_result.get('geographic_hierarchy', {})
                        self.logger.info(f"✅ Smart location mapping completed: {location_result.get('processing_summary', {})}")
                    else:
                        self.logger.warning(f"⚠️ Smart location mapping failed: {location_result.get('error')}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Smart location mapping initialization failed: {e}")
                    # Continue without location mapping
            else:
                with open(debug_path, 'a') as f:
                    f.write(f"Smart location mapping is DISABLED\n")
            
            # Step 2: Process each user through the assignment pipeline (FIRST PASS - Basic Processing)
            # Store users data for location-based assignment
            self._current_users_data = users_data
            
            processed_users = []
            assignment_stats = {
                'total_users': len(users_data),
                'regions_assigned': 0,
                'segments_assigned': 0,
                'modules_assigned': 0,
                'levels_assigned': 0,
                'territories_assigned': 0,
                'areas_assigned': 0,
                'districts_assigned': 0,
                'location_based_assignments': 0
            }
            validation_errors = []
            
            # Debug: Check user processing loop
            with open(debug_path, 'a') as f:
                f.write(f"=== USER PROCESSING LOOP (FIRST PASS) ===\n")
                f.write(f"Total users to process: {len(users_data)}\n")
            
            # FIRST PASS: Process all users with basic field assignments (NO hierarchy analysis)
            for i, user in enumerate(users_data):
                try:
                    with open(debug_path, 'a') as f:
                        f.write(f"Processing user {i+1}/{len(users_data)}: {user.get('Name', 'Unknown')}\n")
                    
                    processed_user = await self._process_single_user_basic(user, config, workflow_config, location_intelligence)
                    
                    with open(debug_path, 'a') as f:
                        f.write(f"Successfully processed user {user.get('Name', 'Unknown')}\n")
                    
                    processed_users.append(processed_user)
                    
                    # Update statistics
                    if processed_user.get('region_assigned'):
                        assignment_stats['regions_assigned'] += 1
                    if processed_user.get('segment_assigned'):
                        assignment_stats['segments_assigned'] += 1
                    if processed_user.get('level_assigned'):
                        assignment_stats['levels_assigned'] += 1
                    if processed_user.get('territory_assigned'):
                        assignment_stats['territories_assigned'] += 1
                    if processed_user.get('area_assigned'):
                        assignment_stats['areas_assigned'] += 1
                    if processed_user.get('district_assigned'):
                        assignment_stats['districts_assigned'] += 1
                    if processed_user.get('location_based'):
                        assignment_stats['location_based_assignments'] += 1
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to process user {user.get('Name', 'Unknown')}: {e}")
                    validation_errors.append({
                        'user': user.get('Name', 'Unknown'),
                        'error': str(e)
                    })
            
            # SECOND PASS: Analyze hierarchy and assign modules based on correct hierarchy
            self.logger.info("🔍 SECOND PASS: Analyzing hierarchy and assigning modules...")
            print("🔍 SECOND PASS: Analyzing hierarchy and assigning modules...")
            
            for i, processed_user in enumerate(processed_users):
                try:
                    email = self._get_field_value(processed_user, 'email')
                    name = self._get_field_value(processed_user, 'name')
                    
                    if email:
                        # Now analyze hierarchy with complete user data
                        has_direct_reports = self._user_has_direct_reports(email, workflow_config)
                        is_leaf_node = self._is_leaf_node_in_hierarchy(email, workflow_config)
                        
                        # Update the processed user with correct hierarchy info
                        processed_user['has_direct_reports'] = has_direct_reports
                        processed_user['is_leaf_node'] = is_leaf_node
                        
                        self.logger.info(f"👥 {name}: has_direct_reports={has_direct_reports}, is_leaf_node={is_leaf_node}")
                        print(f"👥 {name}: has_direct_reports={has_direct_reports}, is_leaf_node={is_leaf_node}")
                        
                        # Now assign modules based on correct hierarchy
                        if config.get('enable_module_assignment', True):
                            try:
                                # Get industry from workflow config (tenant-level setting)
                                industry = workflow_config.get('industry', 'saas')  # Default to SaaS
                                title = self._get_field_value(processed_user, 'title')
                                role_function = self._get_field_value(processed_user, 'department')
                                
                                self.logger.info(f"🎯 Assigning modules for {name} based on correct hierarchy")
                                print(f"🎯 Assigning modules for {name} based on correct hierarchy")
                                
                                # Use centralized module assignment with correct hierarchy
                                assigned_modules = centralized_module_assignment.assign_modules(
                                    user_role=workflow_config.get('default_user_role', 'user'),
                                    job_title=title,
                                    role_function=role_function,
                                    industry=industry,
                                    has_direct_reports=has_direct_reports,  # Now correct!
                                    is_leaf_node=is_leaf_node  # Now correct!
                                )
                                
                                processed_user['Modules'] = ','.join(assigned_modules)
                                processed_user['modules_assigned'] = True
                                assignment_stats['modules_assigned'] += 1
                                
                                self.logger.info(f"✅ {name}: Modules assigned based on correct hierarchy = {assigned_modules}")
                                print(f"✅ {name}: Modules assigned based on correct hierarchy = {assigned_modules}")
                                
                            except Exception as e:
                                self.logger.error(f"❌ ERROR in module assignment for {name}: {str(e)}")
                                print(f"❌ ERROR in module assignment for {name}: {str(e)}")
                                # Fallback to basic modules
                                processed_user['Modules'] = 'Calendar'
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to analyze hierarchy for user {processed_user.get('Name', 'Unknown')}: {e}")
                    validation_errors.append({
                        'user': processed_user.get('Name', 'Unknown'),
                        'error': f"Hierarchy analysis failed: {str(e)}"
                    })
            
            # Calculate confidence scores and statistics
            confidence_scores = self._calculate_confidence_scores(assignment_stats, processed_users)
            
            # Generate confidence statistics
            confidence_results = [{'confidence_result': {
                'overall_confidence': user.get('confidence_score', 0.5),
                'requires_review': user.get('requires_review', False)
            }} for user in processed_users]
            
            confidence_statistics = self.confidence_engine.get_confidence_statistics(confidence_results)
            
            return {
                'success': True,
                'processed_users': processed_users,
                'assignment_statistics': assignment_stats,
                'validation_errors': validation_errors,
                'confidence_scores': confidence_scores,
                'confidence_statistics': confidence_statistics,
                'company_profile': company_profile,
                'geographic_hierarchy': geographic_hierarchy,
                'location_intelligence': location_intelligence,
                'execution_summary': {
                    'total_processed': len(processed_users),
                    'success_rate': (len(processed_users) / len(users_data)) * 100,
                    'location_mapping_enabled': config.get('enable_smart_location_mapping', True),
                    'company_analysis_enabled': True,
                    'confidence_analysis_enabled': True,
                    'avg_confidence_score': confidence_statistics.get('average_confidence', 0.5),
                    'requires_review_count': confidence_statistics.get('requires_review_count', 0),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Onboarding RBA execution failed: {e}")
            return {
                'success': False,
                'error': f"Execution failed: {str(e)}",
                'processed_users': [],
                'assignment_statistics': {}
            }
    
    async def _process_single_user_basic(self, user: Dict[str, Any], config: Dict[str, Any], workflow_config: Dict[str, Any], location_intelligence: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single user through basic field assignment pipeline (NO hierarchy analysis)"""
        processed_user = user.copy()
        location_intelligence = location_intelligence or {}
        
        # DEBUG: Log entry into user processing
        user_name = user.get('Name', user.get('name', 'Unknown'))
        self.logger.info(f"🚀 STARTING _process_single_user_basic for {user_name}")
        print(f"🚀 STARTING _process_single_user_basic for {user_name}")  # Force print to console
        
        # Extract user information using dynamic field mappings
        name = self._get_field_value(user, 'name')
        title = self._get_field_value(user, 'title')  # Maps to JobTitle
        email = self._get_field_value(user, 'email')
        role_function = self._get_field_value(user, 'department')
        
        # PRESERVE MANAGER RELATIONSHIPS FROM CSV
        manager_email = self._get_field_value(user, 'manager_email')
        if manager_email:
            processed_user['Manager Email'] = manager_email
            processed_user['Reports To Email'] = manager_email
            processed_user['Reporting Email'] = manager_email
            self.logger.info(f"👥 Preserved manager relationship for {name}: {manager_email}")
            print(f"👥 Preserved manager relationship for {name}: {manager_email}")
        else:
            self.logger.info(f"👤 No manager for {name} (top-level or root)")
            print(f"👤 No manager for {name} (top-level or root)")
        
        # Debug config to see what's being passed
        with open('debug_config.txt', 'a') as f:
            f.write(f"=== Processing user: {name} ===\n")
            f.write(f"Config: {config}\n")
            f.write(f"enable_region_assignment: {config.get('enable_region_assignment')}\n")
            f.write(f"User data: {user}\n")
            f.write("=" * 50 + "\n")
        
        # Only assign fields that are missing or empty (preserve existing data)
        preserve_existing = config.get('preserve_existing_data', True)
        
        # 1. Region Assignment (with Smart Location Integration)
        if config.get('enable_region_assignment', True):
            # Write debug to file to see what's happening (Windows compatible)
            import os
            debug_path = os.path.join(os.getcwd(), 'debug_region.txt')
            with open(debug_path, 'a') as f:
                f.write(f"Region assignment enabled for {name}\n")
                f.write(f"{name} user data: {user}\n")
                f.write(f"{name} config: {config}\n")
            
            existing_region = user.get('Region', '').strip()
            
            if not preserve_existing or not user.get('Region') or user.get('Region', '').strip() == '':
                with open(debug_path, 'a') as f:
                    f.write(f"Assigning new region for {name}\n")
                
                assigned_region = self._assign_region_smart(name, title, email, config, location_intelligence)
                
                with open(debug_path, 'a') as f:
                    f.write(f"{name} assigned region: {assigned_region}\n")
                
                processed_user['Region'] = assigned_region
                processed_user['region_assigned'] = True
                processed_user['location_based'] = self._is_location_based_assignment(name, email, location_intelligence)
            else:
                with open(debug_path, 'a') as f:
                    f.write(f"Preserving existing region for {name}: {existing_region}\n")
                self.logger.debug(f"👤 {name}: Assigned region = {assigned_region}")
        
        # 2. Level Inference
        if config.get('enable_level_inference', True):
            if not preserve_existing or not user.get('Level') or user.get('Level', '').strip() == '':
                assigned_level = self._infer_level(title, role_function)
                processed_user['Level'] = assigned_level
                processed_user['level_assigned'] = True
                self.logger.debug(f"👤 {name}: Assigned level = {assigned_level}")
        
        # 3. Segment Assignment (depends on level)
        if config.get('enable_segment_assignment', True):
            if not preserve_existing or not user.get('Segment') or user.get('Segment', '').strip() == '':
                user_level = processed_user.get('Level', 'L4')
                assigned_segment = self._assign_segment(title, role_function, user_level)
                processed_user['Segment'] = assigned_segment
                processed_user['segment_assigned'] = True
                self.logger.debug(f"👤 {name}: Assigned segment = {assigned_segment}")
        
        # 4. Territory Assignment (with Smart Location Integration)
        if not preserve_existing or not user.get('Territory') or user.get('Territory', '').strip() == '':
            assigned_territory = self._assign_territory_smart(name, processed_user.get('Region', 'north_america'), location_intelligence)
            processed_user['Territory'] = assigned_territory
            processed_user['territory_assigned'] = True
            self.logger.debug(f"👤 {name}: Assigned territory = {assigned_territory}")
        
        # 4b. Area Assignment (if hierarchical structure enabled)
        if config.get('create_hierarchical_structure', True):
            if not preserve_existing or not user.get('Area') or user.get('Area', '').strip() == '':
                assigned_area = self._assign_area_from_location(name, email, location_intelligence)
                if assigned_area:
                    processed_user['Area'] = assigned_area
                    processed_user['area_assigned'] = True
                    self.logger.debug(f"👤 {name}: Assigned area = {assigned_area}")
        
        # 4c. District Assignment (if hierarchical structure enabled)
        if config.get('create_hierarchical_structure', True):
            if not preserve_existing or not user.get('District') or user.get('District', '').strip() == '':
                assigned_district = self._assign_district_from_location(name, email, location_intelligence)
                if assigned_district:
                    processed_user['District'] = assigned_district
                    processed_user['district_assigned'] = True
                    self.logger.debug(f"👤 {name}: Assigned district = {assigned_district}")
        
        # NOTE: Module assignment is now handled in the SECOND PASS after hierarchy analysis
        
        # Add hierarchical location metadata for frontend (minimal work for frontend)
        if hasattr(self, '_location_results'):
            location_result = self._location_results.get(email)
            if location_result:
                # Add hierarchical admin levels with proper country-specific naming
                admin_levels = location_result.get('admin_levels', {})
                if admin_levels:
                    processed_user['admin_levels'] = admin_levels
                
                # Add territory metadata with human-readable names
                territories = location_result.get('territories', {})
                if territories:
                    processed_user['territory_metadata'] = {
                        'territory_name': territories.get('territory_name'),
                        'territory_type': territories.get('territory_type'),
                        'area_name': territories.get('area_name'),
                        'area_type': territories.get('area_type'),
                        'district_name': territories.get('district_name'),
                        'district_type': territories.get('district_type')
                    }
                
                # Add geolocation metadata
                processed_user['geolocation_metadata'] = {
                    'country_code': location_result.get('country_code'),
                    'detection_method': location_result.get('method'),
                    'confidence': location_result.get('confidence')
                }
        
        # Calculate confidence scores for all assignments
        assignments = {
            'region': processed_user.get('Region'),
            'segment': processed_user.get('Segment'),
            'level': processed_user.get('Level'),
            'territory': processed_user.get('Territory'),
            'area': processed_user.get('Area'),
            'district': processed_user.get('District'),
            'modules': processed_user.get('Modules')
        }
        
        # Filter out None assignments
        assignments = {k: v for k, v in assignments.items() if v is not None}
        
        # Calculate confidence using company profile context
        confidence_result = self.confidence_engine.calculate_assignment_confidence(
            user, assignments, getattr(self, '_current_company_profile', {})
        )
        
        # Add confidence information to processed user
        processed_user['confidence_score'] = confidence_result['overall_confidence']
        processed_user['confidence_level'] = confidence_result['confidence_level']
        processed_user['requires_review'] = confidence_result['requires_review']
        processed_user['field_confidences'] = confidence_result['field_confidences']
        processed_user['low_confidence_fields'] = confidence_result['low_confidence_fields']
        
        # Ensure Name field is set for validation
        processed_user['Name'] = name
        
        return processed_user

    async def _process_single_user(self, user: Dict[str, Any], config: Dict[str, Any], workflow_config: Dict[str, Any], location_intelligence: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single user through the field assignment pipeline"""
        processed_user = user.copy()
        location_intelligence = location_intelligence or {}
        
        # DEBUG: Log entry into user processing
        user_name = user.get('Name', user.get('name', 'Unknown'))
        self.logger.info(f"🚀 STARTING _process_single_user for {user_name}")
        print(f"🚀 STARTING _process_single_user for {user_name}")  # Force print to console
        
        # Extract user information using dynamic field mappings
        name = self._get_field_value(user, 'name')
        title = self._get_field_value(user, 'title')  # Maps to JobTitle
        email = self._get_field_value(user, 'email')
        role_function = self._get_field_value(user, 'department')
        
        # PRESERVE MANAGER RELATIONSHIPS FROM CSV
        manager_email = self._get_field_value(user, 'manager_email')
        if manager_email:
            processed_user['Manager Email'] = manager_email
            processed_user['Reports To Email'] = manager_email
            processed_user['Reporting Email'] = manager_email
            self.logger.info(f"👥 Preserved manager relationship for {name}: {manager_email}")
            print(f"👥 Preserved manager relationship for {name}: {manager_email}")
        else:
            self.logger.info(f"👤 No manager for {name} (top-level or root)")
            print(f"👤 No manager for {name} (top-level or root)")
        
        # Debug config to see what's being passed
        with open('debug_config.txt', 'a') as f:
            f.write(f"=== Processing user: {name} ===\n")
            f.write(f"Config: {config}\n")
            f.write(f"enable_region_assignment: {config.get('enable_region_assignment')}\n")
            f.write(f"User data: {user}\n")
            f.write("=" * 50 + "\n")
        
        # Only assign fields that are missing or empty (preserve existing data)
        preserve_existing = config.get('preserve_existing_data', True)
        
        # 1. Region Assignment (with Smart Location Integration)
        if config.get('enable_region_assignment', True):
            # Write debug to file to see what's happening (Windows compatible)
            import os
            debug_path = os.path.join(os.getcwd(), 'debug_region.txt')
            with open(debug_path, 'a') as f:
                f.write(f"Region assignment enabled for {name}\n")
                f.write(f"{name} user data: {user}\n")
                f.write(f"{name} config: {config}\n")
            
            existing_region = user.get('Region', '').strip()
            
            if not preserve_existing or not user.get('Region') or user.get('Region', '').strip() == '':
                with open(debug_path, 'a') as f:
                    f.write(f"Assigning new region for {name}\n")
                
                assigned_region = self._assign_region_smart(name, title, email, config, location_intelligence)
                
                with open(debug_path, 'a') as f:
                    f.write(f"{name} assigned region: {assigned_region}\n")
                
                processed_user['Region'] = assigned_region
                processed_user['region_assigned'] = True
                processed_user['location_based'] = self._is_location_based_assignment(name, email, location_intelligence)
            else:
                with open(debug_path, 'a') as f:
                    f.write(f"Preserving existing region for {name}: {existing_region}\n")
                self.logger.debug(f"👤 {name}: Assigned region = {assigned_region}")
        
        # 2. Level Inference
        if config.get('enable_level_inference', True):
            if not preserve_existing or not user.get('Level') or user.get('Level', '').strip() == '':
                assigned_level = self._infer_level(title, role_function)
                processed_user['Level'] = assigned_level
                processed_user['level_assigned'] = True
                self.logger.debug(f"👤 {name}: Assigned level = {assigned_level}")
        
        # 3. Segment Assignment (depends on level)
        if config.get('enable_segment_assignment', True):
            if not preserve_existing or not user.get('Segment') or user.get('Segment', '').strip() == '':
                user_level = processed_user.get('Level', 'L4')
                assigned_segment = self._assign_segment(title, role_function, user_level)
                processed_user['Segment'] = assigned_segment
                processed_user['segment_assigned'] = True
                self.logger.debug(f"👤 {name}: Assigned segment = {assigned_segment}")
        
        # 4. Territory Assignment (with Smart Location Integration)
        if not preserve_existing or not user.get('Territory') or user.get('Territory', '').strip() == '':
            assigned_territory = self._assign_territory_smart(name, processed_user.get('Region', 'north_america'), location_intelligence)
            processed_user['Territory'] = assigned_territory
            processed_user['territory_assigned'] = True
            self.logger.debug(f"👤 {name}: Assigned territory = {assigned_territory}")
        
        # 4b. Area Assignment (if hierarchical structure enabled)
        if config.get('create_hierarchical_structure', True):
            if not preserve_existing or not user.get('Area') or user.get('Area', '').strip() == '':
                assigned_area = self._assign_area_from_location(name, email, location_intelligence)
                if assigned_area:
                    processed_user['Area'] = assigned_area
                    processed_user['area_assigned'] = True
                    self.logger.debug(f"👤 {name}: Assigned area = {assigned_area}")
        
        # 4c. District Assignment (if hierarchical structure enabled)
        if config.get('create_hierarchical_structure', True):
            if not preserve_existing or not user.get('District') or user.get('District', '').strip() == '':
                assigned_district = self._assign_district_from_location(name, email, location_intelligence)
                if assigned_district:
                    processed_user['District'] = assigned_district
                    processed_user['district_assigned'] = True
                    self.logger.debug(f"👤 {name}: Assigned district = {assigned_district}")
        
        # NOTE: Module assignment is now handled in the SECOND PASS after hierarchy analysis
        
        # Add hierarchical location metadata for frontend (minimal work for frontend)
        if hasattr(self, '_location_results'):
            location_result = self._location_results.get(email)
            if location_result:
                # Add hierarchical admin levels with proper country-specific naming
                admin_levels = location_result.get('admin_levels', {})
                if admin_levels:
                    processed_user['admin_levels'] = admin_levels
                
                # Add territory metadata with human-readable names
                territories = location_result.get('territories', {})
                if territories:
                    processed_user['territory_metadata'] = {
                        'territory_name': territories.get('territory_name'),
                        'territory_type': territories.get('territory_type'),
                        'area_name': territories.get('area_name'),
                        'area_type': territories.get('area_type'),
                        'district_name': territories.get('district_name'),
                        'district_type': territories.get('district_type')
                    }
                
                # Add geolocation metadata
                processed_user['geolocation_metadata'] = {
                    'country_code': location_result.get('country_code'),
                    'detection_method': location_result.get('method'),
                    'confidence': location_result.get('confidence')
                }
        
        # Calculate confidence scores for all assignments
        assignments = {
            'region': processed_user.get('Region'),
            'segment': processed_user.get('Segment'),
            'level': processed_user.get('Level'),
            'territory': processed_user.get('Territory'),
            'area': processed_user.get('Area'),
            'district': processed_user.get('District'),
            'modules': processed_user.get('Modules')
        }
        
        # Filter out None assignments
        assignments = {k: v for k, v in assignments.items() if v is not None}
        
        # Calculate confidence using company profile context
        confidence_result = self.confidence_engine.calculate_assignment_confidence(
            user, assignments, getattr(self, '_current_company_profile', {})
        )
        
        # Add confidence information to processed user
        processed_user['confidence_score'] = confidence_result['overall_confidence']
        processed_user['confidence_level'] = confidence_result['confidence_level']
        processed_user['requires_review'] = confidence_result['requires_review']
        processed_user['field_confidences'] = confidence_result['field_confidences']
        processed_user['low_confidence_fields'] = confidence_result['low_confidence_fields']
        
        # Ensure Name field is set for validation
        processed_user['Name'] = name
        
        return processed_user
    
    def _assign_region(self, name: str, title: str, email: str, config: Dict[str, Any]) -> str:
        """Assign region based on role analysis and distribution strategy"""
        # Handle None title gracefully
        if not title:
            self.logger.warning(f"⚠️ {name}: No title provided, using default region assignment")
            return self._weighted_random_choice(self.region_weights)
        
        title_lower = title.lower()
        
        # CEO/Chief level always gets Global
        if any(keyword in title_lower for keyword in ['ceo', 'chief', 'president', 'founder']):
            return 'global'
        
        # VP level gets geographic distribution based on keywords
        if any(keyword in title_lower for keyword in ['vp', 'vice president', 'svp', 'senior vice']):
            # Look for geographic hints in title
            if any(geo in title_lower for geo in ['americas', 'north america', 'us', 'usa', 'canada']):
                return 'north_america'
            elif any(geo in title_lower for geo in ['emea', 'europe', 'middle east', 'africa', 'uk', 'germany']):
                return 'emea'
            elif any(geo in title_lower for geo in ['apac', 'asia', 'pacific', 'japan', 'china', 'india']):
                return 'apac'
            elif any(geo in title_lower for geo in ['latam', 'latin america', 'south america', 'brazil', 'mexico']):
                return 'latam'
            
            # No geographic hints, use weighted distribution
            return self._weighted_random_choice(self.region_weights)
        
        # Director level gets mixed distribution
        if any(keyword in title_lower for keyword in ['director', 'head of', 'lead']):
            # Use hash-based distribution for consistency
            return self._hash_based_assignment(name, list(self.region_weights.keys()))
        
        # Individual contributors get weighted random distribution
        return self._weighted_random_choice(self.region_weights)
    
    def _infer_level(self, title: str, role_function: str) -> str:
        """Infer organizational level from title keywords"""
        title_lower = title.lower()
        
        # C-Level mapping
        if any(keyword in title_lower for keyword in ['ceo', 'cto', 'cfo', 'coo', 'cmo', 'chief', 'president', 'founder']):
            return 'L10'
        
        # VP Level mapping
        if any(keyword in title_lower for keyword in ['vp', 'vice president', 'svp', 'senior vice']):
            return 'L9'
        
        # Director Level mapping
        if any(keyword in title_lower for keyword in ['director', 'head of']):
            return 'L8'
        
        # Senior Manager mapping
        if any(keyword in title_lower for keyword in ['senior manager', 'sr manager', 'principal']):
            return 'L7'
        
        # Manager mapping
        if any(keyword in title_lower for keyword in ['manager', 'lead', 'supervisor']):
            return 'L6'
        
        # Senior IC mapping
        if any(keyword in title_lower for keyword in ['senior', 'sr ', 'lead ', 'staff']):
            return 'L5'
        
        # Regular IC mapping
        if any(keyword in title_lower for keyword in ['specialist', 'analyst', 'associate', 'coordinator']):
            return 'L4'
        
        # Junior roles
        if any(keyword in title_lower for keyword in ['junior', 'jr ', 'entry', 'intern', 'trainee']):
            return 'L2'
        
        # Default to mid-level IC
        return 'L4'
    
    def _assign_segment(self, title: str, role_function: str, level: str) -> str:
        """Assign segment based on role and level with realistic distributions"""
        # Determine role category for segment distribution
        title_lower = title.lower()
        
        if level in ['L10']:
            role_category = 'c_level'
        elif level in ['L9']:
            role_category = 'vp_level'
        elif level in ['L8']:
            role_category = 'director'
        elif level in ['L7']:
            role_category = 'senior_manager'
        elif level in ['L6']:
            role_category = 'manager'
        elif level in ['L5']:
            role_category = 'senior_ic'
        else:
            role_category = 'ic'
        
        # Get distribution for this role category
        distribution = self.segment_distributions.get(role_category, self.segment_distributions['ic'])
        
        return self._weighted_random_choice(distribution)
    
    def _assign_territory(self, name: str, region: str) -> str:
        """Assign territory based on region and hash-based distribution"""
        # Territory naming based on region
        territory_prefixes = {
            'north_america': ['NA', 'US', 'CAN'],
            'emea': ['EMEA', 'EU', 'UK'],
            'apac': ['APAC', 'ASIA', 'JP'],
            'latam': ['LATAM', 'BR', 'MX'],
            'global': ['GLOBAL', 'WW', 'INT']
        }
        
        prefixes = territory_prefixes.get(region, ['TERR'])
        prefix = self._hash_based_assignment(name, prefixes)
        
        # Generate territory number (1-99)
        territory_num = (hash(name) % 99) + 1
        
        return f"{prefix}-{territory_num:02d}"
    
    def _user_has_direct_reports(self, email: str, workflow_config: Dict[str, Any]) -> bool:
        """Check if user has direct reports by looking at the CSV data"""
        try:
            # Get all users from the workflow config
            users_data = workflow_config.get('users_data', [])
            
            # Check if any user reports to this email
            for user in users_data:
                manager_email = self._get_field_value(user, 'manager_email')
                if manager_email and manager_email.lower() == email.lower():
                    self.logger.info(f"👥 Found direct report: {self._get_field_value(user, 'name')} reports to {email}")
                    print(f"👥 Found direct report: {self._get_field_value(user, 'name')} reports to {email}")
                    return True
            
            self.logger.info(f"👤 No direct reports found for {email}")
            print(f"👤 No direct reports found for {email}")
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not determine direct reports for {email}: {e}")
            return False
    
    def _is_leaf_node_in_hierarchy(self, email: str, workflow_config: Dict[str, Any]) -> bool:
        """Check if user is a leaf node (IC) in the hierarchy"""
        try:
            # A leaf node is someone who has no direct reports
            has_reports = self._user_has_direct_reports(email, workflow_config)
            is_leaf = not has_reports
            
            if is_leaf:
                self.logger.info(f"🍃 {email} is a leaf node (no direct reports)")
                print(f"🍃 {email} is a leaf node (no direct reports)")
            else:
                self.logger.info(f"🌳 {email} is a manager (has direct reports)")
                print(f"🌳 {email} is a manager (has direct reports)")
            
            return is_leaf
            
        except Exception as e:
            self.logger.warning(f"Could not determine hierarchy position for {email}: {e}")
            return True  # Default to leaf node
    
    def _assign_region_from_location_data(self, name: str, title: str, email: str, config: Dict[str, Any]) -> Optional[str]:
        """Extract region from user's location data using ultra-dynamic hierarchical geolocation service"""
        try:
            # Get the current user data from context
            users_data = getattr(self, '_current_users_data', [])
            print(f"🔍 DEBUG: {name}: Looking for user with email {email} in {len(users_data)} users")
            self.logger.debug(f"🔍 {name}: Looking for user with email {email} in {len(users_data)} users")
            
            current_user = next((u for u in users_data if self._get_field_value(u, 'email') == email), None)
            
            if not current_user:
                print(f"⚠️ DEBUG: {name}: User not found in current context data (email: {email})")
                self.logger.warning(f"⚠️ {name}: User not found in current context data (email: {email})")
                # Debug: show available emails
                available_emails = [self._get_field_value(u, 'email') or 'NO_EMAIL' for u in users_data[:3]]
                print(f"🔍 DEBUG: Available emails (first 3): {available_emails}")
                self.logger.debug(f"🔍 Available emails (first 3): {available_emails}")
                return None
            
            self.logger.debug(f"🔍 {name}: Found user data: {current_user}")
            
            # Use the ultra-dynamic geolocation service to process location data
            location_result = self.geolocation_service.process_user_location(current_user)
            
            self.logger.debug(f"🔍 {name}: Geolocation result: {location_result}")
            
            if location_result['region'] and location_result['confidence'] > 0.5:
                self.logger.info(f"🗺️ {name}: Ultra-dynamic geolocation assigned region = {location_result['region']} "
                               f"(confidence: {location_result['confidence']:.2f}, method: {location_result['method']})")
                
                # Store complete hierarchical location intelligence for territory/area/district assignment
                if not hasattr(self, '_location_results'):
                    self._location_results = {}
                self._location_results[email] = location_result
                
                return location_result['region']
            else:
                self.logger.warning(f"⚠️ {name}: No region detected from location data - region: {location_result.get('region')}, confidence: {location_result.get('confidence')}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error in ultra-dynamic geolocation assignment for {name}: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None
    
    def _assign_region_smart(self, name: str, title: str, email: str, config: Dict[str, Any], location_intelligence: Dict[str, Any]) -> str:
        """Assign region using smart location mapping if available, fallback to original logic"""
        strategy = config.get('region_distribution_strategy', 'smart_location')
        print(f"🔍 DEBUG: _assign_region_smart called for {name}, strategy={strategy}")
        
        # Try smart location mapping first
        if strategy == 'smart_location' and location_intelligence.get('success'):
            print(f"🔍 DEBUG: Trying location intelligence for {name}")
            location_assignments = location_intelligence.get('location_assignments', [])
            print(f"🔍 DEBUG: Total location assignments: {len(location_assignments)}")
            if location_assignments:
                print(f"🔍 DEBUG: First assignment structure: {location_assignments[0]}")
            
            # Try multiple ways to find the user in location assignments
            user_assignment = None
            
            print(f"🔍 DEBUG: Looking for user: name='{name}', email='{email}'")
            
            # Method 1: Try by email (exact match)
            user_assignment = next((la for la in location_assignments if la.get('email') == email), None)
            if user_assignment:
                print(f"✅ DEBUG: Found user by exact email match")
            
            # Method 2: Try by name if email lookup failed (exact match)
            if not user_assignment and name:
                user_assignment = next((la for la in location_assignments if la.get('name') == name), None)
                if user_assignment:
                    print(f"✅ DEBUG: Found user by exact name match")
            
            # Method 3: Try case-insensitive email matching
            if not user_assignment and email:
                user_assignment = next((la for la in location_assignments 
                                      if la.get('email', '').lower() == email.lower()), None)
                if user_assignment:
                    print(f"✅ DEBUG: Found user by case-insensitive email match")
            
            # Method 4: Try case-insensitive name matching
            if not user_assignment and name:
                user_assignment = next((la for la in location_assignments 
                                      if la.get('name', '').lower() == name.lower()), None)
                if user_assignment:
                    print(f"✅ DEBUG: Found user by case-insensitive name match")
            
            # Method 5: Try with dynamic field mapping patterns
            if not user_assignment:
                for la in location_assignments:
                    la_email = la.get('email') or la.get('EmailAddress') or la.get('Email Address', '')
                    la_name = la.get('name') or la.get('Name') or la.get('FullName', '')
                    
                    # Try case-insensitive matching
                    email_match = email and la_email and la_email.lower() == email.lower()
                    name_match = name and la_name and la_name.lower() == name.lower()
                    
                    if email_match or name_match:
                        user_assignment = la
                        print(f"✅ DEBUG: Found user by dynamic field mapping (email_match={email_match}, name_match={name_match})")
                        break
            
            if user_assignment and user_assignment.get('region'):
                print(f"🗺️ DEBUG: {name}: Smart location assigned region = {user_assignment['region']}")
                self.logger.debug(f"🗺️ {name}: Smart location assigned region = {user_assignment['region']}")
                return user_assignment['region']
            else:
                print(f"⚠️ DEBUG: {name}: User not found in location assignments (email: {email})")
                print(f"🔍 DEBUG: Available assignments (first 5): {[{k: v for k, v in la.items() if k in ['name', 'email', 'region']} for la in location_assignments[:5]]}")
                
                # Show all available emails and names for debugging
                available_emails = [la.get('email', 'NO_EMAIL') for la in location_assignments[:10]]
                available_names = [la.get('name', 'NO_NAME') for la in location_assignments[:10]]
                print(f"🔍 DEBUG: Available emails: {available_emails}")
                print(f"🔍 DEBUG: Available names: {available_names}")
                
                self.logger.warning(f"⚠️ {name}: User not found in location assignments")
        
        # NEW: Use advanced geolocation service directly
        if strategy == 'smart_location':
            print(f"🔍 DEBUG: {name}: Trying geolocation service for region assignment...")
            self.logger.debug(f"🔍 {name}: Trying geolocation service for region assignment...")
            region = self._assign_region_from_location_data(name, title, email, config)
            if region:
                print(f"🗺️ DEBUG: {name}: Geolocation service assigned region = {region}")
                self.logger.info(f"🗺️ {name}: Geolocation service assigned region = {region}")
                return region
            else:
                print(f"⚠️ DEBUG: {name}: Geolocation service returned no region")
                self.logger.warning(f"⚠️ {name}: Geolocation service returned no region")
        
        # Fallback to original region assignment logic
        print(f"🔄 DEBUG: {name}: Falling back to original region assignment")
        return self._assign_region(name, title, email, config)
    
    def _assign_territory_smart(self, name: str, region: str, location_intelligence: Dict[str, Any]) -> str:
        """Assign territory using smart location mapping if available"""
        # Try smart location mapping first
        if location_intelligence.get('success'):
            location_assignments = location_intelligence.get('location_assignments', [])
            user_assignment = next((la for la in location_assignments if la.get('name') == name), None)
            
            if user_assignment and user_assignment.get('territory'):
                self.logger.debug(f"🗺️ {name}: Smart location assigned territory = {user_assignment['territory']}")
                return user_assignment['territory']
        
        # NEW: Try ultra-dynamic geolocation service results (hierarchical territories)
        if hasattr(self, '_location_results'):
            for email, location_result in self._location_results.items():
                territories = location_result.get('territories', {})
                if territories.get('territory') and location_result.get('region') == region:
                    territory_code = territories['territory']
                    territory_name = territories.get('territory_name', '')
                    self.logger.info(f"🗺️ {name}: Ultra-dynamic geolocation assigned territory = {territory_code} ({territory_name})")
                    return territory_code
        
        # Fallback to original territory assignment logic
        return self._assign_territory(name, region)
    
    def _assign_area_from_location(self, name: str, email: str, location_intelligence: Dict[str, Any]) -> Optional[str]:
        """Assign area based on ultra-dynamic hierarchical location intelligence"""
        if hasattr(self, '_location_results'):
            location_result = self._location_results.get(email)
            if location_result:
                territories = location_result.get('territories', {})
                if territories.get('area'):
                    area_code = territories['area']
                    area_name = territories.get('area_name', '')
                    area_type = territories.get('area_type', 'Area')
                    self.logger.info(f"🗺️ {name}: Ultra-dynamic assigned area = {area_code} ({area_name}) [{area_type}]")
                    return area_code
        
        return None
    
    def _assign_district_from_location(self, name: str, email: str, location_intelligence: Dict[str, Any]) -> Optional[str]:
        """Assign district based on ultra-dynamic hierarchical location intelligence"""
        if hasattr(self, '_location_results'):
            location_result = self._location_results.get(email)
            if location_result:
                territories = location_result.get('territories', {})
                if territories.get('district'):
                    district_code = territories['district']
                    district_name = territories.get('district_name', '')
                    district_type = territories.get('district_type', 'District')
                    self.logger.info(f"🗺️ {name}: Ultra-dynamic assigned district = {district_code} ({district_name}) [{district_type}]")
                    return district_code
        
        return None
    
    def _is_location_based_assignment(self, name: str, email: str, location_intelligence: Dict[str, Any]) -> bool:
        """Check if user assignment was based on location intelligence"""
        if not location_intelligence.get('success'):
            return False
        
        location_assignments = location_intelligence.get('location_assignments', [])
        user_assignment = next((la for la in location_assignments if la.get('email') == email), None)
        
        return user_assignment is not None and user_assignment.get('location_detected', False)
    
    def _weighted_random_choice(self, weights: Dict[str, float]) -> str:
        """Make a weighted random choice from options"""
        choices = list(weights.keys())
        weights_list = list(weights.values())
        return random.choices(choices, weights=weights_list)[0]
    
    def _hash_based_assignment(self, seed: str, options: List[str]) -> str:
        """Consistent hash-based assignment for deterministic results"""
        hash_value = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        return options[hash_value % len(options)]
    
    def _calculate_confidence_scores(self, stats: Dict[str, Any], processed_users: List[Dict]) -> Dict[str, float]:
        """Calculate confidence scores for assignments"""
        total_users = stats['total_users']
        if total_users == 0:
            return {}
        
        return {
            'region_assignment': (stats['regions_assigned'] / total_users) * 100,
            'segment_assignment': (stats['segments_assigned'] / total_users) * 100,
            'module_assignment': (stats['modules_assigned'] / total_users) * 100,
            'level_inference': (stats['levels_assigned'] / total_users) * 100,
            'territory_assignment': (stats['territories_assigned'] / total_users) * 100,
            'area_assignment': (stats.get('areas_assigned', 0) / total_users) * 100,
            'district_assignment': (stats.get('districts_assigned', 0) / total_users) * 100,
            'location_based_accuracy': (stats.get('location_based_assignments', 0) / total_users) * 100,
            'overall_success': (len(processed_users) / total_users) * 100
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the onboarding RBA agent
        """
        try:
            # Debug: Check if execute is being called
            with open('debug_execute.txt', 'a') as f:
                f.write(f"=== RBA EXECUTE CALLED ===\n")
                f.write(f"Input data keys: {list(input_data.keys())}\n")
                f.write(f"Input data: {input_data}\n")
                f.write("=" * 50 + "\n")
            
            # Extract context and configuration
            context = input_data.get('context', {})
            config = input_data.get('config', {})
            
            # Execute the RBA logic
            result = await self._execute_rba_logic(context, config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ OnboardingRBAAgent execution failed: {e}")
            return {
                'success': False,
                'error': f"Agent execution failed: {str(e)}",
                'processed_users': [],
                'assignment_statistics': {}
            }

    def _initialize_field_mappings(self, users_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize dynamic field mappings based on CSV column names"""
        if not users_data:
            return {}
        
        # Get column names from the first user record
        csv_columns = list(users_data[0].keys())
        
        # Detect field mappings using the dynamic field mapper
        field_mappings = dynamic_field_mapper.detect_field_mappings(csv_columns)
        
        # Log mapping summary
        mapping_summary = dynamic_field_mapper.get_mapping_summary(field_mappings)
        self.logger.info(f"📊 Field mapping summary: {mapping_summary['mapped_fields']}/{mapping_summary['total_standard_fields']} fields mapped "
                        f"(coverage: {mapping_summary['mapping_coverage']:.1%}, avg confidence: {mapping_summary['average_confidence']:.2f})")
        
        # Log individual mappings
        for standard_field, mapping_info in field_mappings.items():
            self.logger.debug(f"🔗 '{mapping_info['csv_column']}' -> '{mapping_info['standard_field']}' (confidence: {mapping_info['confidence']:.2f})")
        
        return field_mappings
    
    def _get_field_value(self, user_data: Dict[str, Any], standard_field: str) -> str:
        """Get field value using dynamic field mappings with fallback to direct access"""
        # Try dynamic mapping first
        if hasattr(self, 'field_mappings') and standard_field in self.field_mappings:
            csv_column = self.field_mappings[standard_field]['csv_column']
            value = user_data.get(csv_column, '')
            if value:
                return str(value).strip()
        
        # Fallback to direct field access (for backward compatibility)
        fallback_mappings = {
            'name': ['Name', 'FullName', 'Full Name', 'Employee Name'],
            'email': ['Email', 'EmailAddress', 'Email Address', 'Work Email'],
            'title': ['Title', 'JobTitle', 'Job Title', 'Role Title', 'Position'],
            'department': ['Department', 'BusinessUnit', 'Business Unit', 'Role Function', 'Team'],
            'manager_name': ['Manager', 'Manager Name', 'Reporting Manager', 'Supervisor'],
            'manager_email': ['Manager Email', 'Reporting Email', 'Reports To Email', 'Supervisor Email'],
            'location': ['Location', 'PrimaryAddress', 'Primary Address', 'Office', 'Address']
        }
        
        if standard_field in fallback_mappings:
            for field_name in fallback_mappings[standard_field]:
                value = user_data.get(field_name, '')
                if value:
                    return str(value).strip()
        
        return ''

def register_agent():
    """Register this agent with the RBA agent registry"""
    try:
        from ...registry.rba_agent_registry import rba_registry
        
        agent_info = {
            'agent_class': OnboardingRBAAgent,
            'category': 'user_management',
            'description': 'Dynamic field assignment for user onboarding',
            'supported_analysis_types': ['onboarding_field_assignment'],
            'priority': 1
        }
        
        rba_registry.register_custom_agent(OnboardingRBAAgent)
        logger.info("✅ OnboardingRBAAgent registered successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to register OnboardingRBAAgent: {e}")

# Auto-register when module is imported
register_agent()
