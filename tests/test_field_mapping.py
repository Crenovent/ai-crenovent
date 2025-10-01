#!/usr/bin/env python3
"""
Test script to verify field mapping with the updated comprehensive patterns
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'hierarchy_processor'))

from hierarchy_processor.core.enhanced_universal_mapper import EnhancedUniversalMapper
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_field_mapping():
    """Test field mapping with the updated patterns"""
    
    # Sample CSV data with the exact field names from user's data
    csv_data = """Employee #,First Name,Last Name,Job Title,Department,Division,Location,Status,Hire Date,Work Email,Reports To,Reports To Email
8753,Diya,Kumar,CEO,Executive,Operations,Pune,Active,09-10-2023,diya.kumar@acme.co,,
6419,Aarav,Chatterjee,CRO,Sales,Product,Mumbai,Active,02-02-2024,aarav.chatterjee@contoso.com,Diya Kumar,diya.kumar@acme.co
6023,Arjun,Pillai,VP Sales,Sales,Sales,Hyderabad,Active,02-08-2021,arjun.pillai@revai.pro,Aarav Chatterjee,aarav.chatterjee@contoso.com"""

    # Write to file
    with open('test_field_mapping.csv', 'w') as f:
        f.write(csv_data)
    
    # Read CSV
    df = pd.read_csv('test_field_mapping.csv')
    
    logger.info(f"📊 Testing Field Mapping:")
    logger.info(f"📊 Input columns: {list(df.columns)}")
    
    # Initialize the enhanced universal mapper
    try:
        mapper = EnhancedUniversalMapper()
        logger.info("✅ Enhanced Universal Mapper initialized successfully")
        
        # Test the mapping
        result_df = mapper.map_any_hrms_to_crenovent(df)
        
        logger.info(f"📊 Output columns: {list(result_df.columns)}")
        logger.info(f"📊 Total records processed: {len(result_df)}")
        
        # Check specific field mappings
        logger.info(f"\n🔍 Field Mapping Results:")
        for idx, row in result_df.iterrows():
            name = row.get('Name', 'Unknown')
            email = row.get('Email', 'N/A')
            reporting_email = row.get('Reporting Email', 'N/A')
            role_title = row.get('Role Title', 'N/A')
            
            logger.info(f"🔍 {name}:")
            logger.info(f"   Email: {email}")
            logger.info(f"   Role Title: {role_title}")
            logger.info(f"   Reporting Email: {reporting_email}")
            logger.info(f"")
        
        # Check field mapping details
        if hasattr(mapper, '_last_field_mapping'):
            logger.info(f"🔧 Field Mapping Details:")
            for field, (column, confidence) in mapper._last_field_mapping.items():
                logger.info(f"   {field} -> {column} ({confidence}% confidence)")
        
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Error in field mapping: {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    test_field_mapping()
