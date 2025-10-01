#!/usr/bin/env python3
"""
Test script to verify the current CSV data structure and email matching
"""
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_csv_structure():
    """Test the structure of the current CSV data"""
    
    # Sample CSV data based on user's provided table
    csv_data = """Employee #,First Name,Last Name,Job Title,Department,Division,Location,Status,Hire Date,Work Email,Reports To,Reports To Email
8753,Diya,Kumar,CEO,Executive,Operations,Pune,Active,09-10-2023,diya.kumar@acme.co,,
6419,Aarav,Chatterjee,CRO,Sales,Product,Mumbai,Active,02-02-2024,aarav.chatterjee@contoso.com,Diya Kumar,diya.kumar@acme.co
6023,Arjun,Pillai,VP Sales,Sales,Sales,Hyderabad,Active,02-08-2021,arjun.pillai@revai.pro,Aarav Chatterjee,aarav.chatterjee@contoso.com
4155,Krishna,Pillai,Finance Analyst,Support,Operations,Delhi,Active,04-05-2020,krishna.pillai@crenovent.com,Arjun Pillai,arjun.pillai@revai.pro
4033,Shaurya,Gupta,Sales Rep,Product,Engineering,Mumbai,Active,05-06-2021,shaurya.gupta@acme.co,Krishna Pillai,krishna.pillai@crenovent.com
5530,Ishaan,Gupta,Sales Rep,Sales,Corporate,Bengaluru,Active,16-02-2020,ishaan.gupta@globex.com,Arjun Pillai,arjun.pillai@revai.pro
5853,Aarav,Saxena,Regional Manager,Finance,Corporate,Hyderabad,Active,18-01-2023,aarav.saxena@revai.pro,Shaurya Gupta,shaurya.gupta@acme.co
5952,Sara,Singh,Director Sales,HR,Corporate,Kolkata,Active,27-12-2024,sara.singh@acme.co,Aarav Saxena,aarav.saxena@revai.pro
5494,Ayaan,Saxena,Director Sales,Marketing,Operations,Kolkata,Active,12-08-2019,ayaan.saxena@contoso.com,Aarav Saxena,aarav.saxena@revai.pro
5786,Atharv,Mehta,Sales Rep,HR,Sales,Pune,Active,03-07-2024,atharv.mehta@contoso.com,Ayaan Saxena,ayaan.saxena@contoso.com
1200,Diya,Singh,Support Specialist,Operations,Engineering,Mumbai,Active,06-04-2024,diya.singh@example.com,Ayaan Saxena,ayaan.saxena@contoso.com
8216,Navya,Mehta,VP Sales,Sales,Sales,Delhi,Active,25-01-2021,navya.mehta@contoso.com,Aarav Chatterjee,aarav.chatterjee@contoso.com
5136,Myra,Bose,QA Engineer,Engineering,Sales,Mumbai,Terminated,12-04-2020,myra.bose@example.com,Arjun Pillai,arjun.pillai@revai.pro
7565,Aarohi,Mehta,Finance Analyst,Legal,Operations,Chennai,Terminated,31-10-2019,aarohi.mehta@acme.co,Ishaan Gupta,ishaan.gupta@globex.com
1878,Myra,Kumar,QA Engineer,HR,Operations,Pune,Active,03-09-2020,myra.kumar@globex.com,Shaurya Gupta,shaurya.gupta@acme.co
5978,Aarohi,Pillai,CRO,HR,Sales,Delhi,Active,13-10-2021,aarohi.pillai@acme.co,Sara Singh,sara.singh@acme.co
7818,Vivaan,Reddy,Regional Manager,Legal,Engineering,Pune,Active,10-11-2024,vivaan.reddy@acme.co,Myra Bose,myra.bose@example.com
8381,Ananya,Sharma,CRO,Engineering,Operations,Bengaluru,Terminated,02-05-2022,ananya.sharma@acme.co,Ayaan Saxena,ayaan.saxena@contoso.com
8025,Krishna,Bose,Finance Analyst,HR,Operations,Pune,Active,18-05-2025,krishna.bose@crenovent.com,Diya Singh,diya.singh@example.com
1986,Ira,Sharma,Regional Manager,Marketing,Corporate,Bengaluru,Active,01-05-2021,ira.sharma@contoso.com,Myra Bose,myra.bose@example.com"""

    # Write to file
    with open('test_hierarchy_debug.csv', 'w') as f:
        f.write(csv_data)
    
    # Read CSV
    df = pd.read_csv('test_hierarchy_debug.csv')
    
    logger.info(f"📊 CSV Structure Analysis:")
    logger.info(f"📊 Total records: {len(df)}")
    logger.info(f"📊 Columns: {list(df.columns)}")
    
    # Analyze email fields
    logger.info(f"\n📧 Email Field Analysis:")
    logger.info(f"📧 Work Email column exists: {'Work Email' in df.columns}")
    logger.info(f"📧 Reports To Email column exists: {'Reports To Email' in df.columns}")
    
    # Sample email data
    logger.info(f"\n📧 Sample Email Data:")
    for idx, row in df.head(5).iterrows():
        work_email = row.get('Work Email', '')
        reports_to_email = row.get('Reports To Email', '')
        name = row.get('First Name', '') + ' ' + row.get('Last Name', '')
        logger.info(f"📧 {name}: Work Email='{work_email}', Reports To Email='{reports_to_email}'")
    
    # Check for hierarchy relationships
    logger.info(f"\n🔗 Hierarchy Analysis:")
    for idx, row in df.iterrows():
        name = row.get('First Name', '') + ' ' + row.get('Last Name', '')
        reports_to_email = row.get('Reports To Email', '')
        
        # Handle NaN values
        if pd.isna(reports_to_email) or reports_to_email == '' or str(reports_to_email).lower() == 'nan':
            reports_to_email = ''
        else:
            reports_to_email = str(reports_to_email).strip()
        
        if reports_to_email:
            # Find the manager
            manager = df[df['Work Email'].str.strip().str.lower() == reports_to_email.lower()]
            if len(manager) > 0:
                manager_name = manager.iloc[0]['First Name'] + ' ' + manager.iloc[0]['Last Name']
                logger.info(f"🔗 {name} -> {manager_name} (via {reports_to_email})")
            else:
                logger.info(f"❌ {name} -> MANAGER NOT FOUND for email '{reports_to_email}'")
                # Debug: Show all available emails
                all_emails = df['Work Email'].tolist()
                logger.info(f"   Available emails: {all_emails[:5]}...")  # Show first 5
        else:
            logger.info(f"🔝 {name} -> TOP LEVEL (no Reports To Email)")
    
    return df

if __name__ == "__main__":
    test_csv_structure()
