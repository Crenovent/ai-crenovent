"""
Smart Location Mapper RBA Agent
Intelligently maps location data from CSV to hierarchical geographic structures.

Features:
- Automatic detection of location columns (address, city, state, country, postal code)
- Geocoding integration for address resolution
- Smart region/area/district/territory assignment
- Head assignment based on location proximity
- Multi-source location data fusion
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent
from .dynamic_field_mapper import dynamic_field_mapper

logger = logging.getLogger(__name__)

class SmartLocationMapperRBAAgent(EnhancedBaseRBAAgent):
    """
    Smart Location Mapper for automatic geographic hierarchy assignment
    """
    
    AGENT_NAME = "smart_location_mapper"
    SUPPORTED_ANALYSIS_TYPES = ["smart_location_mapping", "location_mapping", "geographic_hierarchy"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize geolocation service once
        from .ultra_dynamic_geolocation_service import UltraDynamicGeolocationService
        self.geolocation_service = UltraDynamicGeolocationService()
        
        # Location detection patterns
        self.location_patterns = {
            'address': [
                'address', 'addr', 'street', 'office_location', 'officelocation',
                'work_address', 'business_address', 'mailing_address',
                'physical_address', 'street_address', 'full_address',
                'location', 'primary_address', 'work_location', 'worklocation'
            ],
            'city': [
                'city', 'town', 'municipality', 'locality', 'office_city'
            ],
            'state': [
                'state', 'province', 'region', 'territory', 'prefecture',
                'state_province', 'admin_area'
            ],
            'country': [
                'country', 'nation', 'country_code', 'iso_country'
            ],
            'postal_code': [
                'postal_code', 'zip_code', 'zip', 'postcode', 'postal'
            ],
            'office': [
                'office', 'branch', 'site', 'facility',
                'campus', 'building', 'center', 'office_location', 'officelocation'
            ]
        }
        
        # Patterns to exclude (email-related fields)
        self.exclude_patterns = [
            'email', 'e-mail', 'mail', '@', 'contact'
        ]
        
        # Geographic hierarchy mappings
        self.region_mappings = {
            'north_america': {
                'countries': ['US', 'USA', 'United States', 'CA', 'Canada', 'MX', 'Mexico'],
                'states': ['California', 'Texas', 'New York', 'Florida', 'Ontario', 'Quebec'],
                'keywords': ['america', 'usa', 'canada', 'north', 'na']
            },
            'emea': {
                'countries': ['UK', 'United Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands'],
                'states': ['England', 'Scotland', 'Bavaria', 'Catalonia'],
                'keywords': ['europe', 'emea', 'uk', 'eu', 'middle east', 'africa']
            },
            'apac': {
                'countries': ['Japan', 'China', 'India', 'Australia', 'Singapore', 'South Korea'],
                'states': ['Tokyo', 'Beijing', 'Mumbai', 'Sydney', 'New South Wales'],
                'keywords': ['asia', 'pacific', 'apac', 'japan', 'china', 'australia']
            },
            'latam': {
                'countries': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru'],
                'states': ['São Paulo', 'Buenos Aires', 'Santiago'],
                'keywords': ['latin', 'south america', 'latam', 'brazil', 'argentina']
            }
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the smart location mapping RBA agent
        """
        try:
            # Debug: Check if execute is being called
            import os
            debug_path = os.path.join(os.getcwd(), 'debug_location_intelligence.txt')
            with open(debug_path, 'a') as f:
                f.write(f"=== SmartLocationMapperRBAAgent.execute CALLED ===\n")
                f.write(f"Input data keys: {list(input_data.keys())}\n")
            
            # Extract context and configuration
            context = input_data.get('context', {})
            config = input_data.get('config', {})
            
            # Execute the RBA logic
            result = await self._execute_rba_logic(context, config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ SmartLocationMapperRBAAgent execution failed: {e}")
            return {
                'success': False,
                'error': f"Agent execution failed: {str(e)}",
                'location_intelligence': {},
                'geographic_hierarchy': {}
            }
    
    def _define_config_schema(self) -> Dict[str, Any]:
        """Define configuration schema for location mapping"""
        return {
            'enable_geocoding': {
                'type': 'bool',
                'default': False,
                'description': 'Enable geocoding API for address resolution',
                'ui_component': 'checkbox'
            },
            'auto_detect_locations': {
                'type': 'bool',
                'default': True,
                'description': 'Automatically detect location columns',
                'ui_component': 'checkbox'
            },
            'region_assignment_strategy': {
                'type': 'enum',
                'options': ['country_based', 'keyword_based', 'mixed'],
                'default': 'mixed',
                'description': 'Strategy for region assignment',
                'ui_component': 'dropdown'
            },
            'create_territories': {
                'type': 'bool',
                'default': True,
                'description': 'Automatically create territories based on locations',
                'ui_component': 'checkbox'
            },
            'assign_heads_by_location': {
                'type': 'bool',
                'default': True,
                'description': 'Assign heads based on geographic proximity',
                'ui_component': 'checkbox'
            }
        }
    
    def _define_result_schema(self) -> Dict[str, Any]:
        """Define result schema for location mapping results"""
        return {
            'location_analysis': {
                'type': 'object',
                'description': 'Analysis of detected location data'
            },
            'geographic_hierarchy': {
                'type': 'object',
                'description': 'Generated geographic hierarchy structure'
            },
            'location_assignments': {
                'type': 'array',
                'description': 'Location assignments for each user'
            },
            'detected_columns': {
                'type': 'object',
                'description': 'Detected location-related columns'
            }
        }
    
    async def _execute_rba_logic(self, context: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute smart location mapping logic"""
        try:
            # Debug: Check if _execute_rba_logic is being called
            import os
            debug_path = os.path.join(os.getcwd(), 'debug_location_intelligence.txt')
            with open(debug_path, 'a') as f:
                f.write(f"=== SmartLocationMapperRBAAgent._execute_rba_logic CALLED ===\n")
                f.write(f"Context keys: {list(context.keys())}\n")
                f.write(f"Config: {config}\n")
            
            users_data = context.get('users_data', [])
            if not users_data:
                return {
                    'success': False,
                    'error': 'No user data provided for location mapping'
                }
            
            df = pd.DataFrame(users_data)
            
            # Step 1: Detect location columns
            detected_columns = self._detect_location_columns(df)
            self.logger.info(f"🗺️ Detected location columns: {detected_columns}")
            
            # Step 2: Analyze location data
            location_analysis = self._analyze_location_data(df, detected_columns)
            
            # Step 3: Create geographic hierarchy
            geographic_hierarchy = self._create_geographic_hierarchy(df, detected_columns, config)
            
            # Step 4: Assign locations to users
            location_assignments = self._assign_locations_to_users(df, geographic_hierarchy, config)
            
            # Step 5: Assign heads based on geography (if enabled)
            if config.get('assign_heads_by_location', True):
                location_assignments = self._assign_geographic_heads(location_assignments, df)
            
            return {
                'success': True,
                'location_analysis': location_analysis,
                'geographic_hierarchy': geographic_hierarchy,
                'location_assignments': location_assignments,
                'detected_columns': detected_columns,
                'processing_summary': {
                    'total_users': len(users_data),
                    'users_with_location': len([u for u in location_assignments if u.get('location_detected')]),
                    'regions_created': len(geographic_hierarchy.get('regions', {})),
                    'territories_created': len(geographic_hierarchy.get('territories', {}))
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Smart location mapping failed: {e}")
            return {
                'success': False,
                'error': f"Location mapping failed: {str(e)}"
            }
    
    def _detect_location_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect location-related columns in the DataFrame"""
        detected = {
            'address': [],
            'city': [],
            'state': [],
            'country': [],
            'postal_code': [],
            'office': []
        }
        
        # Create multiple variations of column names for better matching
        for col_idx, original_col in enumerate(df.columns):
            col_variations = [
                original_col.lower(),                           # "Office Location" -> "office location"
                original_col.lower().replace(' ', '_'),         # "Office Location" -> "office_location"
                original_col.lower().replace(' ', ''),          # "Office Location" -> "officelocation"
                original_col.lower().replace('_', ' '),         # "office_location" -> "office location"
                original_col.lower().replace('-', '_'),         # "office-location" -> "office_location"
                original_col.lower().replace('-', ''),          # "office-location" -> "officelocation"
            ]
            
            # Skip columns that match exclude patterns (email-related)
            is_excluded = False
            for exclude_pattern in self.exclude_patterns:
                for col_var in col_variations:
                    if exclude_pattern in col_var:
                        is_excluded = True
                        break
                if is_excluded:
                    break
            
            if is_excluded:
                continue
            
            # Check each location type pattern against all column variations
            for location_type, patterns in self.location_patterns.items():
                matched = False
                for pattern in patterns:
                    for col_var in col_variations:
                        if pattern in col_var:
                            detected[location_type].append(original_col)
                            matched = True
                            break
                    if matched:
                        break
        
        # Remove duplicates
        for key in detected:
            detected[key] = list(set(detected[key]))
        
        # Debug: Log detected columns
        import os
        debug_path = os.path.join(os.getcwd(), 'debug_location_intelligence.txt')
        with open(debug_path, 'a') as f:
            f.write(f"=== COLUMN DETECTION DEBUG ===\n")
            f.write(f"Available columns: {list(df.columns)}\n")
            f.write(f"Detected location columns: {detected}\n")
            for location_type, columns in detected.items():
                if columns:
                    f.write(f"  {location_type}: {columns}\n")
        
        return detected
    
    def _analyze_location_data(self, df: pd.DataFrame, detected_columns: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze the quality and distribution of location data"""
        analysis = {
            'data_quality': {},
            'geographic_distribution': {},
            'coverage_stats': {}
        }
        
        for location_type, columns in detected_columns.items():
            if not columns:
                continue
                
            type_analysis = {
                'columns_found': len(columns),
                'sample_values': [],
                'unique_values': 0,
                'fill_rate': 0
            }
            
            # Analyze first column of each type
            if columns:
                col = columns[0]
                non_null_values = df[col].dropna()
                
                type_analysis['unique_values'] = non_null_values.nunique()
                type_analysis['fill_rate'] = len(non_null_values) / len(df)
                type_analysis['sample_values'] = non_null_values.unique()[:5].tolist()
            
            analysis['data_quality'][location_type] = type_analysis
        
        return analysis
    
    def _create_geographic_hierarchy(self, df: pd.DataFrame, detected_columns: Dict[str, List[str]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create geographic hierarchy based on detected location data"""
        hierarchy = {
            'regions': {},
            'areas': {},
            'districts': {},
            'territories': {}
        }
        
        # Extract location data for analysis
        location_data = []
        for _, row in df.iterrows():
            location_info = self._extract_location_info(row, detected_columns)
            if location_info:
                location_data.append(location_info)
        
        # Create regions based on detected countries/locations
        regions = self._create_regions_from_locations(location_data, config)
        hierarchy['regions'] = regions
        
        # Create territories within regions
        if config.get('create_territories', True):
            territories = self._create_territories_from_locations(location_data, regions)
            hierarchy['territories'] = territories
        
        return hierarchy
    
    def _extract_location_info(self, row: pd.Series, detected_columns: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract location information from a row"""
        location_info = {}
        
        for location_type, columns in detected_columns.items():
            if columns:
                # Use first available column of each type
                col = columns[0]
                value = row.get(col, '')
                if pd.notna(value) and str(value).strip():
                    location_info[location_type] = str(value).strip()
        
        return location_info if location_info else None
    
    def _create_regions_from_locations(self, location_data: List[Dict], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create regions based on location data"""
        regions = {}
        strategy = config.get('region_assignment_strategy', 'mixed')
        
        # Collect all countries and locations
        countries = set()
        cities = set()
        
        for loc in location_data:
            if 'country' in loc:
                countries.add(loc['country'].lower())
            if 'city' in loc:
                cities.add(loc['city'].lower())
        
        # Assign regions based on strategy
        for region_name, region_info in self.region_mappings.items():
            region_countries = set()
            region_cities = set()
            
            # Country-based matching
            for country in countries:
                for region_country in region_info['countries']:
                    if country in region_country.lower() or region_country.lower() in country:
                        region_countries.add(country)
            
            # Keyword-based matching
            if strategy in ['keyword_based', 'mixed']:
                for keyword in region_info['keywords']:
                    for city in cities:
                        if keyword.lower() in city.lower():
                            region_cities.add(city)
            
            if region_countries or region_cities:
                regions[region_name] = {
                    'name': region_name.replace('_', ' ').title(),
                    'code': region_name.upper(),
                    'countries': list(region_countries),
                    'cities': list(region_cities),
                    'territories': []
                }
        
        # Add default region if no matches found
        if not regions:
            regions['global'] = {
                'name': 'Global',
                'code': 'GLOBAL',
                'countries': list(countries),
                'cities': list(cities),
                'territories': []
            }
        
        return regions
    
    def _create_territories_from_locations(self, location_data: List[Dict], regions: Dict[str, Any]) -> Dict[str, Any]:
        """Create territories based on location data and regions"""
        territories = {}
        territory_counter = 1
        
        # First, collect all unique cities/offices from location data
        all_cities = set()
        for loc in location_data:
            # Try multiple location fields
            city = loc.get('city') or loc.get('office') or loc.get('address', 'Unknown')
            if city and city != 'Unknown':
                all_cities.add(city)
        
        print(f"🏙️ DEBUG: Found cities for territory creation: {list(all_cities)}")
        
        for region_name, region_info in regions.items():
            region_territories = []
            
            # Group locations by city/area within region
            city_groups = {}
            for loc in location_data:
                # Extract city from multiple possible fields
                city = loc.get('city') or loc.get('office') or loc.get('address', 'Unknown')
                country = loc.get('country', 'Unknown')
                
                # For North America region, include all US cities
                belongs_to_region = True  # Default to true for now
                
                if region_name == 'north_america':
                    # All US cities belong to north_america
                    us_cities = ['boston', 'san francisco', 'austin', 'new york', 'chicago', 'seattle', 'denver', 'atlanta']
                    belongs_to_region = any(us_city in city.lower() for us_city in us_cities)
                
                if belongs_to_region and city != 'Unknown':
                    if city not in city_groups:
                        city_groups[city] = []
                    city_groups[city].append(loc)
            
            print(f"🏙️ DEBUG: City groups for {region_name}: {list(city_groups.keys())}")
            
            # Create territories from city groups
            for city, locations in city_groups.items():
                territory_code = f"{region_info['code']}-{city.upper().replace(' ', '_')[:3]}-{territory_counter:02d}"
                territory_name = f"{city} Territory"
                
                territories[territory_code] = {
                    'name': territory_name,
                    'code': territory_code,
                    'region': region_name,
                    'city': city,
                    'location_count': len(locations),
                    'primary_city': city
                }
                
                region_territories.append(territory_code)
                territory_counter += 1
                
                print(f"🏙️ DEBUG: Created territory {territory_code} for {city} with {len(locations)} locations")
            
            # Update region with territories
            regions[region_name]['territories'] = region_territories
        
        return territories
    
    def _assign_locations_to_users(self, df: pd.DataFrame, geographic_hierarchy: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assign geographic locations to users"""
        assignments = []
        regions = geographic_hierarchy.get('regions', {})
        territories = geographic_hierarchy.get('territories', {})
        
        # Pre-detect location columns once for the entire DataFrame
        detected_columns = self._detect_location_columns(df)
        
        for idx, row in df.iterrows():
            user_name = self._get_field_value_from_row(row, 'name')
            user_email = self._get_field_value_from_row(row, 'email')
            
            user_assignment = {
                'name': user_name,
                'email': user_email,
                'location_detected': False,
                'region': None,
                'territory': None,
                'location_source': 'inferred'
            }
            
            # Extract location info for this specific user
            location_info = self._extract_location_info(row, detected_columns)
            
            # Write debug to file
            import os
            debug_path = os.path.join(os.getcwd(), 'debug_location_intelligence.txt')
            with open(debug_path, 'a') as f:
                f.write(f"Processing user {user_name} (row {idx})\n")
                f.write(f"Raw row data: {dict(row)}\n")
                f.write(f"Extracted location info: {location_info}\n")
            
            if location_info:
                user_assignment['location_detected'] = True
                user_assignment['raw_location'] = location_info
                
                print(f"🔍 DEBUG: User {user_name} - Calling _assign_user_to_region with location_info: {location_info}")
                
                # Assign region
                assigned_region = self._assign_user_to_region(location_info, regions)
                
                print(f"🔍 DEBUG: User {user_name} - _assign_user_to_region returned: {assigned_region}")
                
                if assigned_region:
                    user_assignment['region'] = assigned_region
                    
                    # Assign territory within region
                    assigned_territory = self._assign_user_to_territory(location_info, territories, assigned_region)
                    if assigned_territory:
                        user_assignment['territory'] = assigned_territory
                else:
                    print(f"⚠️ DEBUG: User {user_name} - No region assigned, defaulting to global")
                    user_assignment['region'] = 'global'
            else:
                print(f"⚠️ DEBUG: User {user_name} - No location info extracted")
                user_assignment['region'] = 'global'
            
            assignments.append(user_assignment)
        
        return assignments
    
    def _assign_user_to_region(self, location_info: Dict[str, Any], regions: Dict[str, Any]) -> Optional[str]:
        """Assign a user to a region using UltraDynamicGeolocationService"""
        try:
            # Write debug to file
            import os
            debug_path = os.path.join(os.getcwd(), 'debug_location_intelligence.txt')
            with open(debug_path, 'a') as f:
                f.write(f"=== _assign_user_to_region START ===\n")
                f.write(f"Input location_info: {location_info}\n")
                f.write(f"Available regions: {list(regions.keys())}\n")
            
            # Create a fresh instance of geolocation service for each call to avoid caching issues
            from .ultra_dynamic_geolocation_service import UltraDynamicGeolocationService
            geolocation_service = UltraDynamicGeolocationService()
            
            # Convert location_info to user data format expected by geolocation service
            user_data = {}
            
            # Map location_info fields to user data fields
            if location_info.get('address'):
                user_data['Address'] = location_info['address']
                user_data['Location'] = location_info['address']
            if location_info.get('office'):
                user_data['Office'] = location_info['office']
            if location_info.get('city'):
                user_data['City'] = location_info['city']
            if location_info.get('state'):
                user_data['State'] = location_info['state']
            if location_info.get('country'):
                user_data['Country'] = location_info['country']
            
            # Build a comprehensive location string
            location_parts = []
            for key in ['address', 'office', 'city', 'state', 'country']:
                if location_info.get(key):
                    location_parts.append(location_info[key])
            
            if location_parts:
                user_data['Location'] = ', '.join(location_parts)
            
            with open(debug_path, 'a') as f:
                f.write(f"Converted user_data: {user_data}\n")
            
            if user_data:
                with open(debug_path, 'a') as f:
                    f.write(f"Calling geolocation service...\n")
                
                result = geolocation_service.process_user_location(user_data)
                
                with open(debug_path, 'a') as f:
                    f.write(f"Geolocation service result: {result}\n")
                
                if result.get('region') and result.get('confidence', 0) > 0.5:
                    with open(debug_path, 'a') as f:
                        f.write(f"SUCCESS: Assigned region '{result['region']}' with confidence {result['confidence']:.2f}\n")
                    print(f"🗺️ Location Intelligence: Assigned region '{result['region']}' with confidence {result['confidence']:.2f}")
                    return result['region']
                else:
                    with open(debug_path, 'a') as f:
                        f.write(f"FAILED: No region or low confidence. Region: {result.get('region')}, Confidence: {result.get('confidence')}\n")
                    print(f"⚠️ Location Intelligence: Failed for {location_info}. Region: {result.get('region')}, Confidence: {result.get('confidence')}")
            else:
                with open(debug_path, 'a') as f:
                    f.write(f"No user_data created from location_info\n")
                print(f"⚠️ Location Intelligence: No user_data created from location_info: {location_info}")
            
            # Fallback to manual region mapping based on location data
            with open(debug_path, 'a') as f:
                f.write(f"Falling back to manual region mapping...\n")
            
            # Enhanced fallback logic with better city/country mapping
            user_country = location_info.get('country', '').lower()
            user_city = location_info.get('city', '').lower()
            user_address = location_info.get('address', '').lower()
            user_office = location_info.get('office', '').lower()
            
            # Combine all location text for analysis
            all_location_text = f"{user_country} {user_city} {user_address} {user_office}".lower()
            
            # Enhanced region mapping based on common patterns
            region_keywords = {
                'north_america': ['usa', 'us', 'america', 'united states', 'canada', 'mexico', 'boston', 'new york', 'san francisco', 'austin', 'chicago', 'toronto', 'vancouver'],
                'emea': ['uk', 'united kingdom', 'britain', 'england', 'germany', 'france', 'italy', 'spain', 'netherlands', 'london', 'berlin', 'paris', 'madrid', 'amsterdam'],
                'apac': ['japan', 'china', 'india', 'australia', 'singapore', 'south korea', 'tokyo', 'beijing', 'mumbai', 'sydney', 'singapore', 'seoul'],
                'latam': ['brazil', 'argentina', 'chile', 'colombia', 'peru', 'sao paulo', 'buenos aires', 'santiago', 'bogota', 'lima']
            }
            
            for region, keywords in region_keywords.items():
                for keyword in keywords:
                    if keyword in all_location_text:
                        with open(debug_path, 'a') as f:
                            f.write(f"Matched region '{region}' based on keyword '{keyword}' in '{all_location_text}'\n")
                        print(f"🗺️ Fallback: Assigned region '{region}' based on keyword '{keyword}'")
                        return region
            
            # Check against existing regions if available
            for region_name, region_info in regions.items():
                # Check country match
                for region_country in region_info.get('countries', []):
                    if user_country in region_country.lower() or region_country.lower() in user_country:
                        with open(debug_path, 'a') as f:
                            f.write(f"Matched region '{region_name}' based on country '{user_country}'\n")
                        return region_name
                
                # Check city match
                for region_city in region_info.get('cities', []):
                    if user_city in region_city.lower() or region_city.lower() in user_city:
                        with open(debug_path, 'a') as f:
                            f.write(f"Matched region '{region_name}' based on city '{user_city}'\n")
                        return region_name
            
            # Default to first region if no match
            default_region = list(regions.keys())[0] if regions else 'global'
            with open(debug_path, 'a') as f:
                f.write(f"No matches found, defaulting to '{default_region}'\n")
            return default_region
            
        except Exception as e:
            print(f"❌ Error in location intelligence region assignment: {e}")
            import traceback
            with open(debug_path, 'a') as f:
                f.write(f"ERROR: {e}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
            # Fallback to 'global' if everything fails
            return 'global'
    
    def _assign_user_to_territory(self, location_info: Dict[str, Any], territories: Dict[str, Any], user_region: str) -> Optional[str]:
        """Assign a user to a territory within their region"""
        # Extract user's city from multiple possible fields
        user_city = (location_info.get('city') or 
                    location_info.get('office') or 
                    location_info.get('address', '')).lower()
        
        print(f"🏙️ DEBUG: Assigning territory for user_city='{user_city}' in region='{user_region}'")
        
        # Find territories in user's region
        region_territories = [t_code for t_code, t_info in territories.items() if t_info.get('region') == user_region]
        print(f"🏙️ DEBUG: Available territories in {user_region}: {region_territories}")
        
        # Try to match by city (exact and partial matching)
        for territory_code in region_territories:
            territory_info = territories[territory_code]
            territory_city = territory_info.get('city', '').lower()
            primary_city = territory_info.get('primary_city', '').lower()
            
            print(f"🏙️ DEBUG: Checking territory {territory_code}: territory_city='{territory_city}', primary_city='{primary_city}'")
            
            # Try multiple matching strategies
            if (user_city and territory_city and 
                (user_city == territory_city or 
                 user_city in territory_city or 
                 territory_city in user_city)):
                print(f"🏙️ DEBUG: Matched territory {territory_code} for city '{user_city}'")
                return territory_code
            
            # Also try primary_city field
            if (user_city and primary_city and 
                (user_city == primary_city or 
                 user_city in primary_city or 
                 primary_city in user_city)):
                print(f"🏙️ DEBUG: Matched territory {territory_code} for city '{user_city}' via primary_city")
                return territory_code
        
        # Assign to first available territory in region
        assigned_territory = region_territories[0] if region_territories else None
        print(f"🏙️ DEBUG: No exact match found, assigning to first available territory: {assigned_territory}")
        return assigned_territory
    
    def _assign_geographic_heads(self, location_assignments: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Assign heads based on geographic proximity and seniority"""
        # This would implement head assignment logic based on location proximity
        # For now, return assignments as-is
        return location_assignments
    
    def _get_field_value_from_row(self, row: pd.Series, standard_field: str) -> str:
        """Get field value from pandas row using dynamic field mappings with fallback"""
        # Convert row to dict for compatibility with dynamic field mapper
        user_data = row.to_dict()
        
        # Fallback mappings for common fields
        fallback_mappings = {
            'name': ['Name', 'FullName', 'Full Name', 'Employee Name'],
            'email': ['Email', 'EmailAddress', 'Email Address', 'Work Email'],
            'title': ['Title', 'JobTitle', 'Job Title', 'Role Title', 'Position'],
            'department': ['Department', 'BusinessUnit', 'Business Unit', 'Role Function', 'Team'],
            'manager_email': ['Manager Email', 'Reporting Email', 'Reports To Email', 'Supervisor Email'],
            'location': ['Location', 'PrimaryAddress', 'Primary Address', 'Office', 'Address']
        }
        
        if standard_field in fallback_mappings:
            for field_name in fallback_mappings[standard_field]:
                value = user_data.get(field_name, '')
                if value and pd.notna(value):
                    return str(value).strip()
        
        return ''

# Register the agent
def register_agent():
    """Register this agent with the RBA agent registry"""
    try:
        from ...registry.enhanced_capability_registry import EnhancedCapabilityRegistry
        rba_registry = EnhancedCapabilityRegistry()
        
        agent_info = {
            'agent_class': SmartLocationMapperRBAAgent,
            'category': 'location_intelligence',
            'description': 'Smart location mapping and geographic hierarchy creation',
            'supported_analysis_types': ['smart_location_mapping'],
            'priority': 1
        }
        
        rba_registry.register_custom_agent(SmartLocationMapperRBAAgent)
        logger.info("✅ SmartLocationMapperRBAAgent registered successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to register SmartLocationMapperRBAAgent: {e}")

# Auto-register when module is imported
register_agent()
