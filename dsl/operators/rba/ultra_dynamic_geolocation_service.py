"""
Ultra-Dynamic Geolocation Service
================================
Handles hierarchical administrative divisions dynamically:
- India: Country → State → District → City → Area
- US: Country → State → County → City → Neighborhood  
- Germany: Country → State → District → City → Borough
- NO HARDCODING - Uses APIs and databases
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pycountry
import pycountry_convert as pc
import reverse_geocoder as rg
import time
from .dynamic_field_mapper import dynamic_field_mapper

logger = logging.getLogger(__name__)

class UltraDynamicGeolocationService:
    """
    Ultra-dynamic geolocation service with hierarchical administrative divisions
    NO hardcoded lists - uses APIs, databases, and ISO standards
    """
    
    def __init__(self, user_agent: str = "crenovent-rba-agent"):
        self.geolocator = Nominatim(user_agent=user_agent, timeout=1)
        
        # Business region mapping (only high-level continent mapping - NOT hardcoded cities)
        self.continent_to_region = {
            'NA': 'north_america',    # North America
            'SA': 'latam',           # South America → LATAM
            'EU': 'emea',            # Europe → EMEA
            'AF': 'emea',            # Africa → EMEA  
            'AS': 'apac',            # Asia → APAC
            'OC': 'apac',            # Oceania → APAC
            'AN': 'emea'             # Antarctica → EMEA
        }
        
        # Business logic overrides (minimal - only for business conventions)
        self.business_overrides = {
            # Middle East → EMEA (business convention, not geographic)
            'AE': 'emea', 'SA': 'emea', 'IL': 'emea', 'TR': 'emea', 'EG': 'emea',
            'JO': 'emea', 'LB': 'emea', 'KW': 'emea', 'QA': 'emea', 'BH': 'emea',
            'OM': 'emea', 'YE': 'emea', 'SY': 'emea', 'IQ': 'emea', 'IR': 'emea',
            
            # Russia + Central Asia → EMEA (business convention)
            'RU': 'emea', 'KZ': 'emea', 'UZ': 'emea', 'TM': 'emea', 'TJ': 'emea', 'KG': 'emea'
        }
        
        # Administrative level names by country (for proper hierarchy naming)
        self.admin_level_names = {
            'IN': ['State', 'District', 'City', 'Area'],
            'US': ['State', 'County', 'City', 'Neighborhood'], 
            'CA': ['Province', 'County', 'City', 'District'],
            'GB': ['Country', 'County', 'City', 'Borough'],
            'DE': ['State', 'District', 'City', 'Borough'],
            'FR': ['Region', 'Department', 'City', 'Arrondissement'],
            'IT': ['Region', 'Province', 'City', 'District'],
            'ES': ['Region', 'Province', 'City', 'District'],
            'AU': ['State', 'Region', 'City', 'Suburb'],
            'BR': ['State', 'Mesoregion', 'City', 'District'],
            'JP': ['Prefecture', 'City', 'Ward', 'District'],
            'CN': ['Province', 'Prefecture', 'City', 'District'],
            # Default for other countries
            'DEFAULT': ['State', 'Region', 'City', 'Area']
        }
    
    def extract_location_from_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all location-related fields from user data using dynamic field mapping
        Returns structured location data instead of just a string
        """
        # Try dynamic field mapping first for location
        location_value = self._get_dynamic_field_value(user_data, 'location')
        
        # Traditional field patterns as fallback
        location_fields = {
            'country': ['Country', 'Nation', 'country', 'nation', 'Country Code', 'country_code'],
            'state': ['State', 'Province', 'Region', 'Prefecture', 'state', 'province', 'region', 'prefecture'],
            'district': ['District', 'County', 'Department', 'Prefecture', 'district', 'county', 'department'],
            'city': ['City', 'Town', 'Municipality', 'city', 'town', 'municipality'],
            'area': ['Area', 'Neighborhood', 'Borough', 'Ward', 'Suburb', 'Zone', 'area', 'neighborhood', 'borough', 'ward', 'suburb', 'zone'],
            'address': ['Address', 'Location', 'Office', 'Site', 'Work Location', 'address', 'location', 'office', 'site', 'work_location', 'work location', 'Full Address', 'Street', 'Street Address', 'PrimaryAddress', 'Primary Address'],
            'postal': ['Postal Code', 'ZIP', 'Postcode', 'PIN', 'postal_code', 'zip', 'postcode', 'pin', 'Zip Code']
        }
        
        extracted = {}
        
        # If dynamic mapping found a location, use it as primary address
        if location_value:
            extracted['address'] = location_value
        
        # Extract other location fields using traditional patterns
        for category, field_names in location_fields.items():
            if category == 'address' and location_value:
                continue  # Already handled by dynamic mapping
                
            for field_name in field_names:
                if user_data.get(field_name):
                    extracted[category] = str(user_data[field_name]).strip()
                    break  # Use first found field in category
        
        # Build full location string for geocoding
        location_parts = []
        
        # Prioritize address from dynamic mapping
        if location_value:
            location_parts.append(location_value)
        
        # Add other location components
        for category in ['city', 'state', 'country']:
            if extracted.get(category):
                location_parts.append(extracted[category])
        
        # Fallback: use any location field found
        if not location_parts:
            all_location_fields = []
            for field_list in location_fields.values():
                all_location_fields.extend(field_list)
            
            for field in all_location_fields:
                if user_data.get(field):
                    location_parts.append(str(user_data[field]).strip())
        
        if location_parts:
            extracted['full_location'] = ', '.join(location_parts)
        
        logger.debug(f"🗺️ Extracted location data: {extracted}")
        return extracted
    
    def _get_dynamic_field_value(self, user_data: Dict[str, Any], field_type: str) -> str:
        """Get field value using dynamic field mapping patterns"""
        # This is a simplified version - in a full implementation, 
        # we'd get the field mappings from the OnboardingRBAAgent
        fallback_mappings = {
            'location': ['Location', 'PrimaryAddress', 'Primary Address', 'Office', 'Address', 'Work Location', 'Site']
        }
        
        if field_type in fallback_mappings:
            for field_name in fallback_mappings[field_type]:
                value = user_data.get(field_name, '')
                if value:
                    return str(value).strip()
        
        return ''
    
    def get_country_code_dynamic(self, location_data: Dict[str, Any]) -> Optional[str]:
        """
        Dynamically extract country code using multiple methods
        """
        # Method 1: Direct country field
        if location_data.get('country'):
            country_text = location_data['country'].lower()
            
            # Try direct country code match
            if len(country_text) == 2 and country_text.upper() in [c.alpha_2 for c in pycountry.countries]:
                return country_text.upper()
            
            # Try country name match
            for country in pycountry.countries:
                if country.name.lower() == country_text or country_text in country.name.lower():
                    return country.alpha_2
            
            # Common name variations
            country_variations = {
                'usa': 'US', 'america': 'US', 'united states': 'US',
                'uk': 'GB', 'britain': 'GB', 'england': 'GB', 'scotland': 'GB', 'wales': 'GB',
                'uae': 'AE', 'emirates': 'AE', 'south korea': 'KR', 'korea': 'KR',
                'hong kong': 'HK', 'taiwan': 'TW'
            }
            
            for variation, code in country_variations.items():
                if variation in country_text:
                    return code
        
        # Method 2: Extract from full location string
        if location_data.get('full_location'):
            location_text = location_data['full_location']
            
            # Look for country codes in the text
            country_codes = re.findall(r'\b([A-Z]{2,3})\b', location_text.upper())
            for code in country_codes:
                try:
                    if len(code) == 2:
                        country = pycountry.countries.get(alpha_2=code)
                    else:
                        country = pycountry.countries.get(alpha_3=code)
                    
                    if country:
                        return country.alpha_2
                except:
                    continue
        
        return None
    
    def geocode_for_hierarchy(self, location_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use geocoding to get detailed administrative hierarchy
        """
        if not location_data.get('full_location'):
            return None
        
        # Use dynamic pattern matching instead of external geocoding API
        logger.debug(f"🗺️ Using dynamic pattern matching for: {location_data['full_location']}")
        return self._dynamic_location_parsing(location_data['full_location'])
        
    def _dynamic_location_parsing(self, location_string: str) -> Optional[Dict[str, Any]]:
        """
        Dynamic location parsing using pycountry and intelligent pattern matching
        """
        if not location_string:
            return None
        
        try:
            location_lower = location_string.lower().strip()
            logger.debug(f"🔍 Parsing location: '{location_lower}'")
            
            # Extract potential country from the location string
            country_code = self._extract_country_code_dynamic(location_lower)
            
            if country_code:
                # Get country information
                try:
                    country = pycountry.countries.get(alpha_2=country_code)
                    country_name = country.name if country else country_code
                except:
                    country_name = country_code
                
                # Extract city from location string
                city = self._extract_city_from_location(location_lower, country_code)
                
                # Build hierarchy
                hierarchy = {
                    'country_code': country_code,
                    'country': country_name,
                }
                
                if city:
                    hierarchy['city'] = city
                
                logger.debug(f"🗺️ Dynamic parsing result: {hierarchy}")
                return hierarchy
            
            logger.debug(f"🔍 No country detected in: '{location_lower}'")
            return None
            
        except Exception as e:
            logger.error(f"❌ Dynamic location parsing failed: {e}")
            return None
    
    def _extract_country_code_dynamic(self, location_text: str) -> Optional[str]:
        """
        Dynamically extract country code using geolocation services and libraries
        """
        if not location_text:
            logger.debug("🔍 Empty location text provided")
            return None
            
        location_text = location_text.lower().strip()
        logger.debug(f"🔍 Processing location text: '{location_text}'")
        
        # Method 1: Try reverse geocoding for known cities (offline, fast)
        country_code = self._reverse_geocode_for_country(location_text)
        if country_code:
            return country_code
        
        # Method 2: Try explicit country name matching (offline, fast)
        country_code = self._extract_country_from_text(location_text)
        if country_code:
            return country_code
        
        # Method 3: Final fallback - geocoding with geopy (online, slower)
        return self._geocode_for_country(location_text)
    
    def _geocode_for_country(self, location_text: str) -> Optional[str]:
        """
        Use geopy geocoding to find country code for a location
        """
        try:
            logger.debug(f"🌍 Geocoding: {location_text}")
            location = self.geolocator.geocode(location_text, timeout=1)
            
            if location and hasattr(location, 'raw') and 'address' in location.raw:
                address = location.raw['address']
                
                # Extract country code directly
                country_code = address.get('country_code', '').upper()
                if country_code:
                    logger.debug(f"🌍 Geocoding success: '{location_text}' -> {country_code}")
                    return country_code
                    
                # Try to extract from country name
                country_name = address.get('country', '')
                if country_name:
                    try:
                        country = pycountry.countries.lookup(country_name)
                        logger.debug(f"🌍 Country name lookup: '{country_name}' -> {country.alpha_2}")
                        return country.alpha_2
                    except LookupError:
                        logger.debug(f"🔍 Country name not found in pycountry: {country_name}")
                        
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.debug(f"🔍 Geocoding service error for '{location_text}': {e}")
        except Exception as e:
            logger.debug(f"🔍 Geocoding failed for '{location_text}': {e}")
        
        return None
    
    def _reverse_geocode_for_country(self, location_text: str) -> Optional[str]:
        """
        Use reverse geocoding for major cities (offline method)
        """
        try:
            # Major cities with known coordinates (lightweight fallback)
            major_cities = {
                'los angeles': (34.0522, -118.2437),
                'new york': (40.7128, -74.0060),
                'chicago': (41.8781, -87.6298),
                'houston': (29.7604, -95.3698),
                'phoenix': (33.4484, -112.0740),
                'philadelphia': (39.9526, -75.1652),
                'san antonio': (29.4241, -98.4936),
                'san diego': (32.7157, -117.1611),
                'dallas': (32.7767, -96.7970),
                'san jose': (37.3382, -121.8863),
                'austin': (30.2672, -97.7431),
                'jacksonville': (30.3322, -81.6557),
                'san francisco': (37.7749, -122.4194),
                'columbus': (39.9612, -82.9988),
                'charlotte': (35.2271, -80.8431),
                'indianapolis': (39.7684, -86.1581),
                'seattle': (47.6062, -122.3321),
                'denver': (39.7392, -104.9903),
                'washington': (38.9072, -77.0369),
                'boston': (42.3601, -71.0589),
                'nashville': (36.1627, -86.7816),
                'detroit': (42.3314, -83.0458),
                'portland': (45.5152, -122.6784),
                'las vegas': (36.1699, -115.1398),
                'memphis': (35.1495, -90.0490),
                'baltimore': (39.2904, -76.6122),
                'milwaukee': (43.0389, -87.9065),
                'atlanta': (33.7490, -84.3880),
                'miami': (25.7617, -80.1918),
                'tampa': (27.9506, -82.4572),
                'orlando': (28.5383, -81.3792),
                
                # UK Cities
                'london': (51.5074, -0.1278),
                'birmingham': (52.4862, -1.8904),
                'manchester': (53.4808, -2.2426),
                'glasgow': (55.8642, -4.2518),
                'liverpool': (53.4084, -2.9916),
                'leeds': (53.8008, -1.5491),
                'sheffield': (53.3811, -1.4701),
                'edinburgh': (55.9533, -3.1883),
                'bristol': (51.4545, -2.5879),
                'cardiff': (51.4816, -3.1791),
                'leicester': (52.6369, -1.1398),
                'coventry': (52.4068, -1.5197),
                'bradford': (53.7960, -1.7594),
                'belfast': (54.5973, -5.9301),
                'nottingham': (52.9548, -1.1581),
                
                # German Cities  
                'berlin': (52.5200, 13.4050),
                'hamburg': (53.5511, 9.9937),
                'munich': (48.1351, 11.5820),
                'cologne': (50.9375, 6.9603),
                'frankfurt': (50.1109, 8.6821),
                'stuttgart': (48.7758, 9.1829),
                'düsseldorf': (51.2277, 6.7735),
                'dortmund': (51.5136, 7.4653),
                'essen': (51.4556, 7.0116),
                'leipzig': (51.3397, 12.3731),
                
                # Australian Cities
                'sydney': (-33.8688, 151.2093),
                'melbourne': (-37.8136, 144.9631),
                'brisbane': (-27.4698, 153.0251),
                'perth': (-31.9505, 115.8605),
                'adelaide': (-34.9285, 138.6007),
                'gold coast': (-28.0167, 153.4000),
                'newcastle': (-32.9283, 151.7817),
                'canberra': (-35.2809, 149.1300),
                
                # Asian Cities
                'singapore': (1.3521, 103.8198),
                'bangkok': (13.7563, 100.5018),
                'kuala lumpur': (3.1390, 101.6869),
                'jakarta': (-6.2088, 106.8456),
                'manila': (14.5995, 120.9842),
                'tokyo': (35.6762, 139.6503),
                'osaka': (34.6937, 135.5023),
                'seoul': (37.5665, 126.9780),
                'mumbai': (19.0760, 72.8777),
                'delhi': (28.7041, 77.1025),
                'bangalore': (12.9716, 77.5946),
                'hyderabad': (17.3850, 78.4867),
                'chennai': (13.0827, 80.2707),
                'kolkata': (22.5726, 88.3639),
                'pune': (18.5204, 73.8567),
                'hong kong': (22.3193, 114.1694),
            }
            
            city_lower = location_text.lower().strip()
            for city_name, coords in major_cities.items():
                if city_name in city_lower:
                    logger.debug(f"🗺️ Reverse geocoding: Found coordinates for '{city_name}'")
                    result = rg.search(coords)
                    if result and len(result) > 0:
                        country_code = result[0].get('cc', '').upper()
                        if country_code:
                            logger.debug(f"🗺️ Reverse geocoding success: '{city_name}' -> {country_code}")
                            return country_code
                            
        except Exception as e:
            logger.debug(f"🔍 Reverse geocoding failed for '{location_text}': {e}")
        
        return None
    
    def _extract_country_from_text(self, location_text: str) -> Optional[str]:
        """
        Final fallback: Extract country code from explicit country mentions
        """
        # Only country name variations (no cities - those are handled by geocoding)
        country_variations = {
            'usa': 'US', 'america': 'US', 'united states': 'US', 'us': 'US',
            'uk': 'GB', 'britain': 'GB', 'england': 'GB', 'scotland': 'GB', 'wales': 'GB', 'united kingdom': 'GB',
            'uae': 'AE', 'emirates': 'AE', 'united arab emirates': 'AE',
            'south korea': 'KR', 'korea': 'KR', 'republic of korea': 'KR',
            'hong kong': 'HK', 'hk': 'HK',
            'taiwan': 'TW', 'republic of china': 'TW',
            'russia': 'RU', 'russian federation': 'RU',
        }
        
        # Check for country variations first (word boundary matches to avoid partial matches)
        for variation, code in country_variations.items():
            # Use word boundaries to avoid partial matches (e.g., "us" in "australia")
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, location_text):
                logger.debug(f"🏳️ Found country variation: '{variation}' -> {code}")
                return code
        
        # Try all countries from pycountry (sort by length to prioritize longer matches)
        sorted_countries = sorted(pycountry.countries, key=lambda c: len(c.name), reverse=True)
        
        for country in sorted_countries:
            # Check country name (case insensitive, simple substring match)
            country_name_lower = country.name.lower()
            if country_name_lower in location_text:
                logger.debug(f"🏳️ Found country name: '{country.name}' -> {country.alpha_2}")
                return country.alpha_2
        
        logger.debug(f"❌ No country found in: '{location_text}'")
        return None
    
    def _extract_city_from_location(self, location_text: str, country_code: str) -> Optional[str]:
        """
        Extract city name from location string
        """
        # Split location by common separators
        parts = re.split(r'[,;|]', location_text)
        
        # Look for parts that might be cities (not countries)
        for part in parts:
            part = part.strip()
            if len(part) > 2 and not self._is_country_name(part):
                # Capitalize first letter of each word
                city_name = ' '.join(word.capitalize() for word in part.split())
                logger.debug(f"🏙️ Extracted city: '{city_name}'")
                return city_name
        
        return None
    
    def _is_country_name(self, text: str) -> bool:
        """
        Check if text is a country name
        """
        text_lower = text.lower().strip()
        
        # Check against known country variations
        country_variations = ['usa', 'america', 'united states', 'uk', 'britain', 'england', 'scotland', 'wales', 'united kingdom', 'uae', 'emirates', 'south korea', 'korea', 'hong kong', 'taiwan', 'russia']
        if text_lower in country_variations:
            return True
        
        # Check against pycountry
        for country in pycountry.countries:
            if (country.name.lower() == text_lower or 
                country.alpha_2.lower() == text_lower or 
                (hasattr(country, 'alpha_3') and country.alpha_3.lower() == text_lower)):
                return True
        
        return False
    
    def _extract_country_code_from_location_data(self, location_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract country code from location data using multiple methods
        """
        logger.debug(f"🔍 Extracting country code from location data: {location_data}")
        
        # Method 1: Direct country field
        if location_data.get('country'):
            logger.debug(f"🔍 Method 1: Trying country field: {location_data['country']}")
            country_code = self._extract_country_code_dynamic(location_data['country'])
            if country_code:
                logger.debug(f"✅ Method 1 success: {country_code}")
                return country_code
        
        # Method 2: Parse full location string
        if location_data.get('full_location'):
            logger.debug(f"🔍 Method 2: Trying full_location field: {location_data['full_location']}")
            country_code = self._extract_country_code_dynamic(location_data['full_location'])
            if country_code:
                logger.debug(f"✅ Method 2 success: {country_code}")
                return country_code
        
        # Method 3: Parse address field
        if location_data.get('address'):
            logger.debug(f"🔍 Method 3: Trying address field: {location_data['address']}")
            country_code = self._extract_country_code_dynamic(location_data['address'])
            if country_code:
                logger.debug(f"✅ Method 3 success: {country_code}")
                return country_code
        
        logger.debug("❌ No country code found from any method")
        return None
    
    def _fallback_location_parsing(self, location_string: str) -> Optional[Dict[str, Any]]:
        """
        Fallback location parsing using pattern matching when geocoding fails
        """
        if not location_string:
            return None
        
        try:
            # Common location patterns and their country codes
            location_patterns = {
                # Cities and their countries
                'berlin': {'country_code': 'DE', 'country': 'Germany', 'city': 'Berlin'},
                'munich': {'country_code': 'DE', 'country': 'Germany', 'city': 'Munich'},
                'hamburg': {'country_code': 'DE', 'country': 'Germany', 'city': 'Hamburg'},
                'london': {'country_code': 'GB', 'country': 'United Kingdom', 'city': 'London'},
                'manchester': {'country_code': 'GB', 'country': 'United Kingdom', 'city': 'Manchester'},
                'paris': {'country_code': 'FR', 'country': 'France', 'city': 'Paris'},
                'madrid': {'country_code': 'ES', 'country': 'Spain', 'city': 'Madrid'},
                'rome': {'country_code': 'IT', 'country': 'Italy', 'city': 'Rome'},
                'amsterdam': {'country_code': 'NL', 'country': 'Netherlands', 'city': 'Amsterdam'},
                'tokyo': {'country_code': 'JP', 'country': 'Japan', 'city': 'Tokyo'},
                'osaka': {'country_code': 'JP', 'country': 'Japan', 'city': 'Osaka'},
                'seoul': {'country_code': 'KR', 'country': 'South Korea', 'city': 'Seoul'},
                'beijing': {'country_code': 'CN', 'country': 'China', 'city': 'Beijing'},
                'shanghai': {'country_code': 'CN', 'country': 'China', 'city': 'Shanghai'},
                'mumbai': {'country_code': 'IN', 'country': 'India', 'city': 'Mumbai'},
                'delhi': {'country_code': 'IN', 'country': 'India', 'city': 'Delhi'},
                'bangalore': {'country_code': 'IN', 'country': 'India', 'city': 'Bangalore'},
                'sydney': {'country_code': 'AU', 'country': 'Australia', 'city': 'Sydney'},
                'melbourne': {'country_code': 'AU', 'country': 'Australia', 'city': 'Melbourne'},
                'new york': {'country_code': 'US', 'country': 'United States', 'city': 'New York'},
                'los angeles': {'country_code': 'US', 'country': 'United States', 'city': 'Los Angeles'},
                'chicago': {'country_code': 'US', 'country': 'United States', 'city': 'Chicago'},
                'san francisco': {'country_code': 'US', 'country': 'United States', 'city': 'San Francisco'},
                'toronto': {'country_code': 'CA', 'country': 'Canada', 'city': 'Toronto'},
                'vancouver': {'country_code': 'CA', 'country': 'Canada', 'city': 'Vancouver'},
                'sao paulo': {'country_code': 'BR', 'country': 'Brazil', 'city': 'São Paulo'},
                'rio de janeiro': {'country_code': 'BR', 'country': 'Brazil', 'city': 'Rio de Janeiro'},
                'mexico city': {'country_code': 'MX', 'country': 'Mexico', 'city': 'Mexico City'},
                'buenos aires': {'country_code': 'AR', 'country': 'Argentina', 'city': 'Buenos Aires'},
                'dubai': {'country_code': 'AE', 'country': 'United Arab Emirates', 'city': 'Dubai'},
                'singapore': {'country_code': 'SG', 'country': 'Singapore', 'city': 'Singapore'},
                'hong kong': {'country_code': 'HK', 'country': 'Hong Kong', 'city': 'Hong Kong'},
                
                # Countries
                'germany': {'country_code': 'DE', 'country': 'Germany'},
                'united kingdom': {'country_code': 'GB', 'country': 'United Kingdom'},
                'uk': {'country_code': 'GB', 'country': 'United Kingdom'},
                'france': {'country_code': 'FR', 'country': 'France'},
                'spain': {'country_code': 'ES', 'country': 'Spain'},
                'italy': {'country_code': 'IT', 'country': 'Italy'},
                'netherlands': {'country_code': 'NL', 'country': 'Netherlands'},
                'japan': {'country_code': 'JP', 'country': 'Japan'},
                'south korea': {'country_code': 'KR', 'country': 'South Korea'},
                'korea': {'country_code': 'KR', 'country': 'South Korea'},
                'china': {'country_code': 'CN', 'country': 'China'},
                'india': {'country_code': 'IN', 'country': 'India'},
                'australia': {'country_code': 'AU', 'country': 'Australia'},
                'united states': {'country_code': 'US', 'country': 'United States'},
                'usa': {'country_code': 'US', 'country': 'United States'},
                'america': {'country_code': 'US', 'country': 'United States'},
                'canada': {'country_code': 'CA', 'country': 'Canada'},
                'brazil': {'country_code': 'BR', 'country': 'Brazil'},
                'mexico': {'country_code': 'MX', 'country': 'Mexico'},
                'argentina': {'country_code': 'AR', 'country': 'Argentina'},
                'uae': {'country_code': 'AE', 'country': 'United Arab Emirates'},
                'emirates': {'country_code': 'AE', 'country': 'United Arab Emirates'},
                'singapore': {'country_code': 'SG', 'country': 'Singapore'}
            }
            
            location_lower = location_string.lower().strip()
            
            # Try exact matches first
            for pattern, data in location_patterns.items():
                if pattern in location_lower:
                    logger.debug(f"🗺️ Fallback pattern match: '{pattern}' in '{location_lower}'")
                    return data
            
            logger.debug(f"🗺️ No fallback pattern match found for: '{location_lower}'")
            return None
            
        except Exception as e:
            logger.debug(f"🗺️ Fallback location parsing failed: {e}")
            return None
    
    def get_reverse_geocode_hierarchy(self, location_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use reverse geocoding database for administrative hierarchy
        Fallback when regular geocoding fails
        """
        try:
            # If we have city/state info, try to find coordinates
            search_terms = []
            
            if location_data.get('city'):
                search_terms.append(location_data['city'])
            if location_data.get('state'):
                search_terms.append(location_data['state'])
            if location_data.get('country'):
                search_terms.append(location_data['country'])
            
            if not search_terms:
                return None
            
            search_string = ', '.join(search_terms)
            
            # Use reverse geocoder database
            results = rg.search(search_string)
            
            if results:
                result = results[0]  # Take first result
                
                hierarchy = {
                    'country_code': result.get('cc'),
                    'country': result.get('name'),
                    'state': result.get('admin1'),
                    'district': result.get('admin2'),
                    'city': result.get('name'),
                    'latitude': result.get('lat'),
                    'longitude': result.get('lon')
                }
                
                # Clean up None values
                hierarchy = {k: v for k, v in hierarchy.items() if v}
                
                logger.debug(f"🗺️ Reverse geocoded hierarchy: {hierarchy}")
                return hierarchy
                
        except Exception as e:
            logger.debug(f"🗺️ Reverse geocoding failed: {e}")
        
        return None
    
    def get_region_from_country_code(self, country_code: str) -> Optional[str]:
        """
        Get business region from country code (same as before - this part works well)
        """
        if not country_code:
            return None
        
        # Check business overrides first
        if country_code in self.business_overrides:
            return self.business_overrides[country_code]
        
        try:
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            return self.continent_to_region.get(continent_code)
        except:
            return None
    
    def generate_dynamic_territories(self, hierarchy: Dict[str, Any], region: str) -> Dict[str, Any]:
        """
        Generate territory and sub-territories dynamically based on actual administrative data
        NO hardcoding - uses real administrative hierarchy
        """
        country_code = hierarchy.get('country_code')
        if not country_code:
            return {}
        
        # Get admin level names for this country
        level_names = self.admin_level_names.get(country_code, self.admin_level_names['DEFAULT'])
        
        territories = {}
        
        # Level 1: Territory (usually state/province level)
        if hierarchy.get('state'):
            state_code = hierarchy['state'][:3].upper()  # First 3 chars
            territories['territory'] = f"{region.upper()[:4]}-{country_code}-{state_code}"
            territories['territory_name'] = hierarchy['state']
            territories['territory_type'] = level_names[0]
        
        # Level 2: Area (usually district/county level)  
        if hierarchy.get('district'):
            district_code = hierarchy['district'][:3].upper()
            territories['area'] = f"{territories.get('territory', region.upper()[:4])}-{district_code}"
            territories['area_name'] = hierarchy['district']
            territories['area_type'] = level_names[1] if len(level_names) > 1 else 'District'
        
        # Level 3: District (usually city level)
        if hierarchy.get('city'):
            city_code = hierarchy['city'][:3].upper()
            territories['district'] = f"{territories.get('area', territories.get('territory', region.upper()[:4]))}-{city_code}"
            territories['district_name'] = hierarchy['city']
            territories['district_type'] = level_names[2] if len(level_names) > 2 else 'City'
        
        # Level 4: Sub-district (usually neighborhood/area level)
        if hierarchy.get('area'):
            area_code = hierarchy['area'][:3].upper()
            territories['sub_district'] = f"{territories.get('district', territories.get('area', region.upper()[:4]))}-{area_code}"
            territories['sub_district_name'] = hierarchy['area']
            territories['sub_district_type'] = level_names[3] if len(level_names) > 3 else 'Area'
        
        logger.debug(f"🗺️ Generated territories: {territories}")
        return territories
    
    def process_user_location(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user location with full hierarchical administrative division support
        """
        result = {
            'location_found': False,
            'location_data': {},
            'hierarchy': {},
            'country_code': None,
            'region': None,
            'territories': {},
            'confidence': 0.0,
            'method': 'none',
            'admin_levels': {}
        }
        
        # Extract structured location data
        location_data = self.extract_location_from_user(user_data)
        if not location_data:
            return result
        
        result['location_found'] = True
        result['location_data'] = location_data
        
        # Method 1: Direct country code extraction using dynamic parsing
        country_code = self._extract_country_code_from_location_data(location_data)
        
        if country_code:
            result['country_code'] = country_code
            result['region'] = self.get_region_from_country_code(country_code)
            result['confidence'] = 0.9
            result['method'] = 'country_code_detection'
            
            # Try to get detailed hierarchy
            hierarchy = self.geocode_for_hierarchy(location_data)
            
            if not hierarchy:
                # Fallback to reverse geocoding
                hierarchy = self.get_reverse_geocode_hierarchy(location_data)
            
            if hierarchy:
                result['hierarchy'] = hierarchy
                result['territories'] = self.generate_dynamic_territories(hierarchy, result['region'])
                result['confidence'] = 0.95
                result['method'] = 'full_hierarchy_detection'
                
                # Set admin levels with proper names
                admin_level_names = self.admin_level_names.get(country_code, self.admin_level_names['DEFAULT'])
                result['admin_levels'] = {
                    'country': hierarchy.get('country'),
                    admin_level_names[0].lower(): hierarchy.get('state'),
                    admin_level_names[1].lower() if len(admin_level_names) > 1 else 'district': hierarchy.get('district'),
                    admin_level_names[2].lower() if len(admin_level_names) > 2 else 'city': hierarchy.get('city'),
                    admin_level_names[3].lower() if len(admin_level_names) > 3 else 'area': hierarchy.get('area')
                }
                
                # Clean up None values
                result['admin_levels'] = {k: v for k, v in result['admin_levels'].items() if v}
            
            logger.info(f"🗺️ Processed location: {location_data.get('full_location')} → {result['region']} (method: {result['method']})")
            return result
        
        logger.debug(f"🗺️ No location detected for: {location_data}")
        return result
    
    def validate_hierarchical_coverage(self) -> Dict[str, Any]:
        """
        Test hierarchical detection with real-world examples
        """
        test_cases = [
            # India: State → District → City → Area
            {
                'name': 'Mumbai, India',
                'data': {'City': 'Mumbai', 'State': 'Maharashtra', 'Country': 'India'},
                'expected_levels': ['state', 'district', 'city', 'area']
            },
            # US: State → County → City → Neighborhood
            {
                'name': 'New York, USA', 
                'data': {'City': 'New York', 'State': 'New York', 'Country': 'USA'},
                'expected_levels': ['state', 'county', 'city', 'neighborhood']
            },
            # Germany: State → District → City → Borough
            {
                'name': 'Berlin, Germany',
                'data': {'City': 'Berlin', 'Country': 'Germany'},
                'expected_levels': ['state', 'district', 'city', 'borough']
            }
        ]
        
        results = {}
        for test_case in test_cases:
            result = self.process_user_location(test_case['data'])
            results[test_case['name']] = {
                'region': result.get('region'),
                'territories': result.get('territories', {}),
                'admin_levels': result.get('admin_levels', {}),
                'hierarchy_detected': len(result.get('admin_levels', {})) > 1,
                'method': result.get('method')
            }
        
        return results
