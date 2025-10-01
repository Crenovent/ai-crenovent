"""
Reliable Geolocation Service
Uses offline databases and reliable methods instead of external APIs
"""

import logging
from typing import Dict, List, Any, Optional
import pycountry
import pycountry_convert as pc
import re

logger = logging.getLogger(__name__)

class ReliableGeolocationService:
    """
    Provides reliable geolocation services without external API dependencies
    Uses comprehensive offline databases for city-country mapping
    """
    
    def __init__(self):
        self.continent_to_region_map = {
            'NA': 'north_america',
            'SA': 'latam', 
            'AS': 'apac',
            'EU': 'emea',
            'AF': 'emea',  # Business decision: Africa is part of EMEA
            'OC': 'apac',
            'AN': 'global'  # Antarctica
        }
        
        # Business overrides for specific countries
        self.country_region_overrides = {
            'AE': 'emea', 'SA': 'emea', 'QA': 'emea', 'KW': 'emea', 'BH': 'emea', 'OM': 'emea',  # Middle East
            'RU': 'emea', 'TR': 'emea', 'IL': 'emea', 'ZA': 'emea',  # Business convention
            'EG': 'emea', 'MA': 'emea', 'TN': 'emea', 'DZ': 'emea', 'LY': 'emea',  # North Africa
        }
        
        # Comprehensive city-country database (major cities worldwide)
        self.city_country_database = {
            # North America
            'new york': 'US', 'los angeles': 'US', 'chicago': 'US', 'houston': 'US', 'phoenix': 'US',
            'philadelphia': 'US', 'san antonio': 'US', 'san diego': 'US', 'dallas': 'US', 'san jose': 'US',
            'austin': 'US', 'jacksonville': 'US', 'fort worth': 'US', 'columbus': 'US', 'charlotte': 'US',
            'san francisco': 'US', 'indianapolis': 'US', 'seattle': 'US', 'denver': 'US', 'washington': 'US',
            'boston': 'US', 'el paso': 'US', 'detroit': 'US', 'nashville': 'US', 'portland': 'US',
            'memphis': 'US', 'oklahoma city': 'US', 'las vegas': 'US', 'louisville': 'US', 'baltimore': 'US',
            'milwaukee': 'US', 'albuquerque': 'US', 'tucson': 'US', 'fresno': 'US', 'mesa': 'US',
            'sacramento': 'US', 'atlanta': 'US', 'kansas city': 'US', 'colorado springs': 'US', 'miami': 'US',
            'raleigh': 'US', 'omaha': 'US', 'long beach': 'US', 'virginia beach': 'US', 'oakland': 'US',
            
            'toronto': 'CA', 'montreal': 'CA', 'vancouver': 'CA', 'calgary': 'CA', 'edmonton': 'CA',
            'ottawa': 'CA', 'winnipeg': 'CA', 'quebec city': 'CA', 'hamilton': 'CA', 'kitchener': 'CA',
            
            'mexico city': 'MX', 'guadalajara': 'MX', 'monterrey': 'MX', 'puebla': 'MX', 'tijuana': 'MX',
            'leon': 'MX', 'juarez': 'MX', 'torreon': 'MX', 'merida': 'MX', 'mexicali': 'MX',
            
            # Europe (EMEA)
            'london': 'GB', 'birmingham': 'GB', 'manchester': 'GB', 'glasgow': 'GB', 'liverpool': 'GB',
            'leeds': 'GB', 'sheffield': 'GB', 'edinburgh': 'GB', 'bristol': 'GB', 'cardiff': 'GB',
            
            'berlin': 'DE', 'hamburg': 'DE', 'munich': 'DE', 'cologne': 'DE', 'frankfurt': 'DE',
            'stuttgart': 'DE', 'dusseldorf': 'DE', 'dortmund': 'DE', 'essen': 'DE', 'leipzig': 'DE',
            
            'paris': 'FR', 'marseille': 'FR', 'lyon': 'FR', 'toulouse': 'FR', 'nice': 'FR',
            'nantes': 'FR', 'strasbourg': 'FR', 'montpellier': 'FR', 'bordeaux': 'FR', 'lille': 'FR',
            
            'madrid': 'ES', 'barcelona': 'ES', 'valencia': 'ES', 'seville': 'ES', 'zaragoza': 'ES',
            'malaga': 'ES', 'murcia': 'ES', 'palma': 'ES', 'las palmas': 'ES', 'bilbao': 'ES',
            
            'rome': 'IT', 'milan': 'IT', 'naples': 'IT', 'turin': 'IT', 'palermo': 'IT',
            'genoa': 'IT', 'bologna': 'IT', 'florence': 'IT', 'bari': 'IT', 'catania': 'IT',
            
            'amsterdam': 'NL', 'rotterdam': 'NL', 'the hague': 'NL', 'utrecht': 'NL', 'eindhoven': 'NL',
            'tilburg': 'NL', 'groningen': 'NL', 'almere': 'NL', 'breda': 'NL', 'nijmegen': 'NL',
            
            'zurich': 'CH', 'geneva': 'CH', 'basel': 'CH', 'bern': 'CH', 'lausanne': 'CH',
            'winterthur': 'CH', 'lucerne': 'CH', 'st. gallen': 'CH', 'lugano': 'CH', 'biel': 'CH',
            
            'vienna': 'AT', 'graz': 'AT', 'linz': 'AT', 'salzburg': 'AT', 'innsbruck': 'AT',
            
            'stockholm': 'SE', 'gothenburg': 'SE', 'malmo': 'SE', 'uppsala': 'SE', 'vasteras': 'SE',
            
            'oslo': 'NO', 'bergen': 'NO', 'trondheim': 'NO', 'stavanger': 'NO', 'drammen': 'NO',
            
            'copenhagen': 'DK', 'aarhus': 'DK', 'odense': 'DK', 'aalborg': 'DK', 'esbjerg': 'DK',
            
            'helsinki': 'FI', 'espoo': 'FI', 'tampere': 'FI', 'vantaa': 'FI', 'oulu': 'FI',
            
            'brussels': 'BE', 'antwerp': 'BE', 'ghent': 'BE', 'charleroi': 'BE', 'liege': 'BE',
            
            'dublin': 'IE', 'cork': 'IE', 'limerick': 'IE', 'galway': 'IE', 'waterford': 'IE',
            
            'lisbon': 'PT', 'porto': 'PT', 'amadora': 'PT', 'braga': 'PT', 'setubal': 'PT',
            
            'athens': 'GR', 'thessaloniki': 'GR', 'patras': 'GR', 'heraklion': 'GR', 'larissa': 'GR',
            
            'warsaw': 'PL', 'krakow': 'PL', 'lodz': 'PL', 'wroclaw': 'PL', 'poznan': 'PL',
            'gdansk': 'PL', 'szczecin': 'PL', 'bydgoszcz': 'PL', 'lublin': 'PL', 'katowice': 'PL',
            
            'prague': 'CZ', 'brno': 'CZ', 'ostrava': 'CZ', 'plzen': 'CZ', 'liberec': 'CZ',
            
            'budapest': 'HU', 'debrecen': 'HU', 'szeged': 'HU', 'miskolc': 'HU', 'pecs': 'HU',
            
            'bucharest': 'RO', 'cluj-napoca': 'RO', 'timisoara': 'RO', 'iasi': 'RO', 'constanta': 'RO',
            
            'sofia': 'BG', 'plovdiv': 'BG', 'varna': 'BG', 'burgas': 'BG', 'ruse': 'BG',
            
            'moscow': 'RU', 'saint petersburg': 'RU', 'novosibirsk': 'RU', 'yekaterinburg': 'RU',
            'nizhny novgorod': 'RU', 'kazan': 'RU', 'chelyabinsk': 'RU', 'omsk': 'RU', 'samara': 'RU',
            
            'istanbul': 'TR', 'ankara': 'TR', 'izmir': 'TR', 'bursa': 'TR', 'adana': 'TR',
            'gaziantep': 'TR', 'konya': 'TR', 'antalya': 'TR', 'kayseri': 'TR', 'mersin': 'TR',
            
            # Middle East & Africa (EMEA)
            'dubai': 'AE', 'abu dhabi': 'AE', 'sharjah': 'AE', 'al ain': 'AE', 'ajman': 'AE',
            
            'riyadh': 'SA', 'jeddah': 'SA', 'mecca': 'SA', 'medina': 'SA', 'dammam': 'SA',
            'khobar': 'SA', 'tabuk': 'SA', 'buraidah': 'SA', 'khamis mushait': 'SA', 'hail': 'SA',
            
            'doha': 'QA', 'al rayyan': 'QA', 'umm salal': 'QA', 'al wakrah': 'QA', 'al khor': 'QA',
            
            'kuwait city': 'KW', 'hawalli': 'KW', 'salmiya': 'KW', 'sabah al salem': 'KW', 'jahra': 'KW',
            
            'manama': 'BH', 'riffa': 'BH', 'muharraq': 'BH', 'hamad town': 'BH', 'isa town': 'BH',
            
            'muscat': 'OM', 'seeb': 'OM', 'salalah': 'OM', 'bawshar': 'OM', 'sohar': 'OM',
            
            'tel aviv': 'IL', 'jerusalem': 'IL', 'haifa': 'IL', 'rishon lezion': 'IL', 'petah tikva': 'IL',
            'ashdod': 'IL', 'netanya': 'IL', 'beer sheva': 'IL', 'holon': 'IL', 'bnei brak': 'IL',
            
            'cairo': 'EG', 'alexandria': 'EG', 'giza': 'EG', 'shubra el kheima': 'EG', 'port said': 'EG',
            'suez': 'EG', 'luxor': 'EG', 'mansoura': 'EG', 'el mahalla el kubra': 'EG', 'tanta': 'EG',
            
            'casablanca': 'MA', 'rabat': 'MA', 'fez': 'MA', 'marrakech': 'MA', 'agadir': 'MA',
            'tangier': 'MA', 'meknes': 'MA', 'oujda': 'MA', 'kenitra': 'MA', 'tetouan': 'MA',
            
            'tunis': 'TN', 'sfax': 'TN', 'sousse': 'TN', 'kairouan': 'TN', 'bizerte': 'TN',
            
            'algiers': 'DZ', 'oran': 'DZ', 'constantine': 'DZ', 'annaba': 'DZ', 'blida': 'DZ',
            
            'tripoli': 'LY', 'benghazi': 'LY', 'misrata': 'LY', 'tarhuna': 'LY', 'khoms': 'LY',
            
            'cape town': 'ZA', 'johannesburg': 'ZA', 'durban': 'ZA', 'pretoria': 'ZA', 'port elizabeth': 'ZA',
            'pietermaritzburg': 'ZA', 'benoni': 'ZA', 'tembisa': 'ZA', 'east london': 'ZA', 'vereeniging': 'ZA',
            
            'lagos': 'NG', 'kano': 'NG', 'ibadan': 'NG', 'abuja': 'NG', 'port harcourt': 'NG',
            'benin city': 'NG', 'maiduguri': 'NG', 'zaria': 'NG', 'aba': 'NG', 'jos': 'NG',
            
            'nairobi': 'KE', 'mombasa': 'KE', 'nakuru': 'KE', 'eldoret': 'KE', 'kisumu': 'KE',
            
            'addis ababa': 'ET', 'dire dawa': 'ET', 'mekelle': 'ET', 'gondar': 'ET', 'awasa': 'ET',
            
            # Asia-Pacific (APAC)
            'tokyo': 'JP', 'yokohama': 'JP', 'osaka': 'JP', 'nagoya': 'JP', 'sapporo': 'JP',
            'fukuoka': 'JP', 'kobe': 'JP', 'kawasaki': 'JP', 'kyoto': 'JP', 'saitama': 'JP',
            'hiroshima': 'JP', 'sendai': 'JP', 'kitakyushu': 'JP', 'chiba': 'JP', 'sakai': 'JP',
            
            'seoul': 'KR', 'busan': 'KR', 'incheon': 'KR', 'daegu': 'KR', 'daejeon': 'KR',
            'gwangju': 'KR', 'suwon': 'KR', 'ulsan': 'KR', 'changwon': 'KR', 'goyang': 'KR',
            
            'beijing': 'CN', 'shanghai': 'CN', 'guangzhou': 'CN', 'shenzhen': 'CN', 'tianjin': 'CN',
            'wuhan': 'CN', 'dongguan': 'CN', 'chengdu': 'CN', 'nanjing': 'CN', 'foshan': 'CN',
            'shenyang': 'CN', 'qingdao': 'CN', 'xian': 'CN', 'zibo': 'CN', 'zhengzhou': 'CN',
            'hangzhou': 'CN', 'suzhou': 'CN', 'harbin': 'CN', 'jinan': 'CN', 'dalian': 'CN',
            
            'hong kong': 'HK', 'kowloon': 'HK', 'tsuen wan': 'HK', 'yuen long': 'HK', 'tuen mun': 'HK',
            
            'taipei': 'TW', 'kaohsiung': 'TW', 'taichung': 'TW', 'tainan': 'TW', 'taoyuan': 'TW',
            
            'singapore': 'SG',
            
            'kuala lumpur': 'MY', 'george town': 'MY', 'ipoh': 'MY', 'shah alam': 'MY', 'petaling jaya': 'MY',
            'klang': 'MY', 'johor bahru': 'MY', 'subang jaya': 'MY', 'seremban': 'MY', 'kuala terengganu': 'MY',
            
            'bangkok': 'TH', 'samut prakan': 'TH', 'mueang nonthaburi': 'TH', 'udon thani': 'TH', 'chon buri': 'TH',
            'nakhon ratchasima': 'TH', 'chiang mai': 'TH', 'hat yai': 'TH', 'ubon ratchathani': 'TH', 'khon kaen': 'TH',
            
            'jakarta': 'ID', 'surabaya': 'ID', 'bandung': 'ID', 'bekasi': 'ID', 'medan': 'ID',
            'tangerang': 'ID', 'depok': 'ID', 'semarang': 'ID', 'palembang': 'ID', 'makassar': 'ID',
            
            'manila': 'PH', 'quezon city': 'PH', 'caloocan': 'PH', 'davao': 'PH', 'cebu city': 'PH',
            'zamboanga': 'PH', 'antipolo': 'PH', 'pasig': 'PH', 'taguig': 'PH', 'valenzuela': 'PH',
            
            'ho chi minh city': 'VN', 'hanoi': 'VN', 'hai phong': 'VN', 'da nang': 'VN', 'bien hoa': 'VN',
            'hue': 'VN', 'nha trang': 'VN', 'can tho': 'VN', 'rach gia': 'VN', 'qui nhon': 'VN',
            
            'mumbai': 'IN', 'delhi': 'IN', 'bangalore': 'IN', 'hyderabad': 'IN', 'ahmedabad': 'IN',
            'chennai': 'IN', 'kolkata': 'IN', 'surat': 'IN', 'pune': 'IN', 'jaipur': 'IN',
            'lucknow': 'IN', 'kanpur': 'IN', 'nagpur': 'IN', 'indore': 'IN', 'thane': 'IN',
            'bhopal': 'IN', 'visakhapatnam': 'IN', 'pimpri chinchwad': 'IN', 'patna': 'IN', 'vadodara': 'IN',
            'ghaziabad': 'IN', 'ludhiana': 'IN', 'agra': 'IN', 'nashik': 'IN', 'faridabad': 'IN',
            'meerut': 'IN', 'rajkot': 'IN', 'kalyan dombivali': 'IN', 'vasai virar': 'IN', 'varanasi': 'IN',
            
            'karachi': 'PK', 'lahore': 'PK', 'faisalabad': 'PK', 'rawalpindi': 'PK', 'gujranwala': 'PK',
            'peshawar': 'PK', 'multan': 'PK', 'hyderabad': 'PK', 'islamabad': 'PK', 'quetta': 'PK',
            
            'dhaka': 'BD', 'chittagong': 'BD', 'khulna': 'BD', 'rajshahi': 'BD', 'sylhet': 'BD',
            'barisal': 'BD', 'rangpur': 'BD', 'comilla': 'BD', 'narayanganj': 'BD', 'gazipur': 'BD',
            
            'colombo': 'LK', 'dehiwala mount lavinia': 'LK', 'moratuwa': 'LK', 'sri jayawardenepura kotte': 'LK', 'negombo': 'LK',
            
            'sydney': 'AU', 'melbourne': 'AU', 'brisbane': 'AU', 'perth': 'AU', 'adelaide': 'AU',
            'gold coast': 'AU', 'newcastle': 'AU', 'canberra': 'AU', 'sunshine coast': 'AU', 'wollongong': 'AU',
            'geelong': 'AU', 'hobart': 'AU', 'townsville': 'AU', 'cairns': 'AU', 'darwin': 'AU',
            
            'auckland': 'NZ', 'wellington': 'NZ', 'christchurch': 'NZ', 'hamilton': 'NZ', 'tauranga': 'NZ',
            'napier hastings': 'NZ', 'dunedin': 'NZ', 'palmerston north': 'NZ', 'nelson': 'NZ', 'rotorua': 'NZ',
            
            # Latin America (LATAM)
            'sao paulo': 'BR', 'rio de janeiro': 'BR', 'brasilia': 'BR', 'salvador': 'BR', 'fortaleza': 'BR',
            'belo horizonte': 'BR', 'manaus': 'BR', 'curitiba': 'BR', 'recife': 'BR', 'porto alegre': 'BR',
            'goiania': 'BR', 'belem': 'BR', 'guarulhos': 'BR', 'campinas': 'BR', 'sao luis': 'BR',
            'sao goncalo': 'BR', 'maceio': 'BR', 'duque de caxias': 'BR', 'natal': 'BR', 'teresina': 'BR',
            
            'buenos aires': 'AR', 'cordoba': 'AR', 'rosario': 'AR', 'mendoza': 'AR', 'tucuman': 'AR',
            'la plata': 'AR', 'mar del plata': 'AR', 'salta': 'AR', 'santa fe': 'AR', 'san juan': 'AR',
            
            'santiago': 'CL', 'valparaiso': 'CL', 'concepcion': 'CL', 'la serena': 'CL', 'antofagasta': 'CL',
            'temuco': 'CL', 'rancagua': 'CL', 'talca': 'CL', 'arica': 'CL', 'chilian': 'CL',
            
            'bogota': 'CO', 'medellin': 'CO', 'cali': 'CO', 'barranquilla': 'CO', 'cartagena': 'CO',
            'cucuta': 'CO', 'bucaramanga': 'CO', 'pereira': 'CO', 'santa marta': 'CO', 'ibague': 'CO',
            
            'lima': 'PE', 'arequipa': 'PE', 'trujillo': 'PE', 'chiclayo': 'PE', 'piura': 'PE',
            'iquitos': 'PE', 'cusco': 'PE', 'chimbote': 'PE', 'huancayo': 'PE', 'tacna': 'PE',
            
            'caracas': 'VE', 'maracaibo': 'VE', 'valencia': 'VE', 'barquisimeto': 'VE', 'maracay': 'VE',
            'ciudad guayana': 'VE', 'san cristobal': 'VE', 'maturin': 'VE', 'ciudad bolivar': 'VE', 'cumana': 'VE',
            
            'quito': 'EC', 'guayaquil': 'EC', 'cuenca': 'EC', 'santo domingo': 'EC', 'machala': 'EC',
            
            'la paz': 'BO', 'santa cruz': 'BO', 'cochabamba': 'BO', 'sucre': 'BO', 'oruro': 'BO',
            
            'asuncion': 'PY', 'ciudad del este': 'PY', 'san lorenzo': 'PY', 'luque': 'PY', 'capiata': 'PY',
            
            'montevideo': 'UY', 'salto': 'UY', 'paysandu': 'UY', 'las piedras': 'UY', 'rivera': 'UY',
            
            'panama city': 'PA', 'san miguelito': 'PA', 'tocumen': 'PA', 'david': 'PA', 'arraijan': 'PA',
            
            'san jose': 'CR', 'cartago': 'CR', 'puntarenas': 'CR', 'alajuela': 'CR', 'heredia': 'CR',
            
            'guatemala city': 'GT', 'mixco': 'GT', 'villa nueva': 'GT', 'petapa': 'GT', 'san juan sacatepequez': 'GT',
            
            'tegucigalpa': 'HN', 'san pedro sula': 'HN', 'choloma': 'HN', 'la ceiba': 'HN', 'el progreso': 'HN',
            
            'managua': 'NI', 'leon': 'NI', 'masaya': 'NI', 'matagalpa': 'NI', 'chinandega': 'NI',
            
            'san salvador': 'SV', 'soyapango': 'SV', 'santa ana': 'SV', 'san miguel': 'SV', 'mejicanos': 'SV',
            
            'havana': 'CU', 'santiago de cuba': 'CU', 'camaguey': 'CU', 'holguin': 'CU', 'guantanamo': 'CU',
            
            'santo domingo': 'DO', 'santiago': 'DO', 'san pedro de macoris': 'DO', 'la romana': 'DO', 'san francisco de macoris': 'DO',
            
            'port au prince': 'HT', 'cap haitien': 'HT', 'carrefour': 'HT', 'delmas': 'HT', 'gonaives': 'HT',
            
            'kingston': 'JM', 'spanish town': 'JM', 'portmore': 'JM', 'montego bay': 'JM', 'may pen': 'JM',
        }
        
        # Country name variations and codes
        self.country_variations = {
            'usa': 'US', 'america': 'US', 'united states': 'US', 'united states of america': 'US',
            'uk': 'GB', 'britain': 'GB', 'great britain': 'GB', 'united kingdom': 'GB', 
            'england': 'GB', 'scotland': 'GB', 'wales': 'GB', 'northern ireland': 'GB',
            'uae': 'AE', 'emirates': 'AE', 'united arab emirates': 'AE',
            'south korea': 'KR', 'korea': 'KR', 'republic of korea': 'KR',
            'north korea': 'KP', 'dprk': 'KP',
            'hong kong': 'HK', 'hk': 'HK',
            'taiwan': 'TW', 'republic of china': 'TW',
            'russia': 'RU', 'russian federation': 'RU',
            'iran': 'IR', 'islamic republic of iran': 'IR',
            'syria': 'SY', 'syrian arab republic': 'SY',
            'venezuela': 'VE', 'bolivarian republic of venezuela': 'VE',
            'bolivia': 'BO', 'plurinational state of bolivia': 'BO',
            'tanzania': 'TZ', 'united republic of tanzania': 'TZ',
            'congo': 'CG', 'republic of the congo': 'CG',
            'drc': 'CD', 'democratic republic of the congo': 'CD', 'congo drc': 'CD',
            'czech republic': 'CZ', 'czechia': 'CZ',
            'slovakia': 'SK', 'slovak republic': 'SK',
            'macedonia': 'MK', 'north macedonia': 'MK',
            'bosnia': 'BA', 'bosnia and herzegovina': 'BA',
            'serbia': 'RS', 'republic of serbia': 'RS',
            'montenegro': 'ME', 'republic of montenegro': 'ME',
            'moldova': 'MD', 'republic of moldova': 'MD',
        }
        
        logger.info("🌍 ReliableGeolocationService initialized with comprehensive offline database.")
    
    def process_user_location(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to process user location data and return region/territory information
        """
        try:
            # Extract location data
            location_data = self._extract_location_data(user_data)
            
            if not location_data:
                return self._create_empty_result()
            
            # Get country code using reliable methods
            country_code = self._get_country_code_reliable(location_data)
            
            if not country_code:
                return self._create_empty_result()
            
            # Get business region from country code
            region = self._get_business_region(country_code)
            
            # Generate territories and hierarchy
            territories = self._generate_territories(country_code, location_data, region)
            admin_levels = self._get_admin_level_names(country_code)
            
            return {
                'location_found': True,
                'location_data': location_data,
                'hierarchy': {
                    'country_code': country_code,
                    'country': self._get_country_name(country_code),
                    'city': location_data.get('city'),
                    'state': location_data.get('state'),
                },
                'country_code': country_code,
                'region': region,
                'territories': territories,
                'confidence': 0.95,  # High confidence for reliable offline data
                'method': 'reliable_offline_database',
                'admin_levels': admin_levels
            }
            
        except Exception as e:
            logger.error(f"❌ Error in reliable geolocation processing: {e}")
            return self._create_empty_result()
    
    def _extract_location_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location data from user record"""
        location_fields = {
            'country': ['Country', 'Nation', 'country', 'nation', 'Country Code', 'country_code'],
            'state': ['State', 'Province', 'Region', 'Prefecture', 'state', 'province', 'region', 'prefecture'],
            'city': ['City', 'Town', 'Municipality', 'city', 'town', 'municipality'],
            'address': ['Address', 'Location', 'Office', 'Site', 'Work Location', 'address', 'location', 'office', 'site', 'work_location', 'work location', 'Full Address', 'Street', 'Street Address'],
        }
        
        extracted = {}
        
        for category, field_names in location_fields.items():
            for field_name in field_names:
                if user_data.get(field_name):
                    extracted[category] = str(user_data[field_name]).strip()
                    break
        
        # Create full location string
        location_parts = []
        for field in ['address', 'city', 'state', 'country']:
            if extracted.get(field):
                location_parts.append(extracted[field])
        
        if location_parts:
            extracted['full_location'] = ', '.join(location_parts)
        
        return extracted
    
    def _get_country_code_reliable(self, location_data: Dict[str, Any]) -> Optional[str]:
        """Get country code using reliable offline methods"""
        
        # Method 1: Direct country field
        if location_data.get('country'):
            country_text = location_data['country'].lower().strip()
            
            # Try direct country code match (2-letter)
            if len(country_text) == 2 and country_text.upper() in [c.alpha_2 for c in pycountry.countries]:
                return country_text.upper()
            
            # Try country name variations
            if country_text in self.country_variations:
                return self.country_variations[country_text]
            
            # Try pycountry lookup
            try:
                country = pycountry.countries.lookup(country_text)
                return country.alpha_2
            except:
                pass
        
        # Method 2: City lookup in comprehensive database
        if location_data.get('city'):
            city_name = location_data['city'].lower().strip()
            if city_name in self.city_country_database:
                country_code = self.city_country_database[city_name]
                logger.debug(f"🏙️ City '{city_name}' mapped to country '{country_code}'")
                return country_code
        
        # Method 3: Parse full location string
        if location_data.get('full_location'):
            full_text = location_data['full_location'].lower()
            
            # Look for country names in the text
            for country_name, country_code in self.country_variations.items():
                if country_name in full_text:
                    return country_code
            
            # Look for cities in the text
            for city_name, country_code in self.city_country_database.items():
                if city_name in full_text:
                    return country_code
        
        return None
    
    def _get_business_region(self, country_code: str) -> str:
        """Map country code to business region"""
        
        # Check business overrides first
        if country_code in self.country_region_overrides:
            return self.country_region_overrides[country_code]
        
        # Use continent mapping
        try:
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            return self.continent_to_region_map.get(continent_code, 'global')
        except:
            return 'global'
    
    def _generate_territories(self, country_code: str, location_data: Dict[str, Any], region: str) -> Dict[str, Any]:
        """Generate territory codes and names"""
        territories = {}
        
        # Generate territory code: REGION-COUNTRY
        territory_code = f"{region.upper()}-{country_code}"
        country_name = self._get_country_name(country_code)
        
        territories['territory'] = territory_code
        territories['territory_name'] = country_name
        territories['territory_type'] = 'Country'
        
        # Add state/province level if available
        if location_data.get('state'):
            state_code = location_data['state'][:3].upper()
            state_territory = f"{territory_code}-{state_code}"
            territories['area'] = state_territory
            territories['area_name'] = location_data['state']
            territories['area_type'] = self._get_state_type_name(country_code)
        
        # Add city level if available
        if location_data.get('city'):
            city_code = location_data['city'][:3].upper()
            if location_data.get('state'):
                city_territory = f"{territories['area']}-{city_code}"
            else:
                city_territory = f"{territory_code}-{city_code}"
            territories['district'] = city_territory
            territories['district_name'] = location_data['city']
            territories['district_type'] = 'City'
        
        return territories
    
    def _get_country_name(self, country_code: str) -> str:
        """Get country name from country code"""
        try:
            country = pycountry.countries.get(alpha_2=country_code)
            return country.name if country else country_code
        except:
            return country_code
    
    def _get_state_type_name(self, country_code: str) -> str:
        """Get the appropriate name for state/province level"""
        state_names = {
            'US': 'State', 'CA': 'Province', 'AU': 'State', 'IN': 'State',
            'DE': 'State', 'BR': 'State', 'AR': 'Province', 'MX': 'State',
            'CN': 'Province', 'JP': 'Prefecture', 'FR': 'Region', 'IT': 'Region',
            'ES': 'Community', 'GB': 'Country', 'RU': 'Oblast'
        }
        return state_names.get(country_code, 'State')
    
    def _get_admin_level_names(self, country_code: str) -> Dict[str, str]:
        """Get country-specific administrative level names"""
        admin_names = {
            'US': {'country': 'United States', 'state': 'State', 'city': 'City'},
            'CA': {'country': 'Canada', 'state': 'Province', 'city': 'City'},
            'GB': {'country': 'United Kingdom', 'state': 'Country', 'city': 'City'},
            'DE': {'country': 'Germany', 'state': 'State', 'city': 'City'},
            'FR': {'country': 'France', 'state': 'Region', 'city': 'City'},
            'IN': {'country': 'India', 'state': 'State', 'city': 'City'},
            'JP': {'country': 'Japan', 'state': 'Prefecture', 'city': 'City'},
            'CN': {'country': 'China', 'state': 'Province', 'city': 'City'},
            'AU': {'country': 'Australia', 'state': 'State', 'city': 'City'},
            'BR': {'country': 'Brazil', 'state': 'State', 'city': 'City'},
        }
        
        default = {'country': self._get_country_name(country_code), 'state': 'State', 'city': 'City'}
        return admin_names.get(country_code, default)
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no location data is found"""
        return {
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
